from .errors import *
from .utils import make_str, log
from . import settings

import io
import json
import hashlib
import math
import requests.utils
import time
import re
import datetime as dt
import os
import logging as lg
from collections import OrderedDict
from dateutil import parser as date_parser


def get_osm_filter(network_type):

    filters = {}

    # driving: filter out un-drivable roads, service roads, private ways, and
    # anything specifying motor=no. also filter out any non-service roads that
    # are tagged as providing parking, driveway, private, or emergency-access
    # services
    filters['drive'] = ('["area"!~"yes"]["highway"!~"cycleway|footway|path|pedestrian|steps|track|corridor|'
                        'elevator|escalator|proposed|construction|bridleway|abandoned|platform|raceway|service"]'
                        '["motor_vehicle"!~"no"]["motorcar"!~"no"]{}'
                        '["service"!~"parking|parking_aisle|driveway|private|emergency_access"]').format(
        settings.default_access)

    # drive+service: allow ways tagged 'service' but filter out certain types of
    # service ways
    filters['drive_service'] = ('["area"!~"yes"]["highway"!~"cycleway|footway|path|pedestrian|steps|track|corridor|'
                                'elevator|escalator|proposed|construction|bridleway|abandoned|platform|raceway"]'
                                '["motor_vehicle"!~"no"]["motorcar"!~"no"]{}'
                                '["service"!~"parking|parking_aisle|private|emergency_access"]').format(
        settings.default_access)

    # walking: filter out cycle ways, motor ways, private ways, and anything
    # specifying foot=no. allow service roads, permitting things like parking
    # lot lanes, alleys, etc that you *can* walk on even if they're not exactly
    # pleasant walks. some cycleways may allow pedestrians, but this filter ignores
    # such cycleways.
    filters['walk'] = ('["area"!~"yes"]["highway"!~"cycleway|motor|proposed|construction|abandoned|platform|raceway"]'
                       '["foot"!~"no"]["service"!~"private"]{}').format(settings.default_access)

    # biking: filter out foot ways, motor ways, private ways, and anything
    # specifying biking=no
    filters['bike'] = ('["area"!~"yes"]["highway"!~"footway|steps|corridor|elevator|escalator|motor|proposed|'
                       'construction|abandoned|platform|raceway"]'
                       '["bicycle"!~"no"]["service"!~"private"]{}').format(settings.default_access)

    # to download all ways, just filter out everything not currently in use or
    # that is private-access only
    filters['all'] = ('["area"!~"yes"]["highway"!~"proposed|construction|abandoned|platform|raceway"]'
                      '["service"!~"private"]{}').format(settings.default_access)

    # to download all ways, including private-access ones, just filter out
    # everything not currently in use
    filters['all_private'] = '["area"!~"yes"]["highway"!~"proposed|construction|abandoned|platform|raceway"]'

    # no filter, needed for infrastructures other than "highway"
    filters['none'] = ''

    if network_type in filters:
        osm_filter = filters[network_type]
    else:
        raise UnknownNetworkType('unknown network_type "{}"'.format(network_type))

    return osm_filter


def save_to_cache(url, response_json):

    if settings.use_cache:
        if response_json is None:
            log('Saved nothing to cache because response_json is None')
        else:
            # create the folder on the disk if it doesn't already exist
            if not os.path.exists(settings.cache_folder):
                os.makedirs(settings.cache_folder)

            # hash the url (to make filename shorter than the often extremely
            # long url)
            filename = hashlib.md5(url.encode('utf-8')).hexdigest()
            cache_path_filename = os.path.join(settings.cache_folder, os.extsep.join([filename, 'json']))

            # dump to json, and save to file
            json_str = make_str(json.dumps(response_json))
            with io.open(cache_path_filename, 'w', encoding='utf-8') as cache_file:
                cache_file.write(json_str)

            log('Saved response to cache file "{}"'.format(cache_path_filename))


def get_from_cache(url):

    # if the tool is configured to use the cache
    if settings.use_cache:
        # determine the filename by hashing the url
        filename = hashlib.md5(url.encode('utf-8')).hexdigest()

        cache_path_filename = os.path.join(settings.cache_folder, os.extsep.join([filename, 'json']))
        # open the cache file for this url hash if it already exists, otherwise
        # return None
        if os.path.isfile(cache_path_filename):
            with io.open(cache_path_filename, encoding='utf-8') as cache_file:
                response_json = json.load(cache_file)
            log('Retrieved response from cache file "{}" for URL "{}"'.format(cache_path_filename, url))
            return response_json


def get_http_headers(user_agent=None, referer=None, accept_language=None):

    if user_agent is None:
        user_agent = settings.default_user_agent
    if referer is None:
        referer = settings.default_referer
    if accept_language is None:
        accept_language = settings.default_accept_language

    headers = requests.utils.default_headers()
    headers.update({'User-Agent': user_agent, 'referer': referer, 'Accept-Language': accept_language})
    return headers


def get_pause_duration(recursive_delay=5, default_duration=10):

    try:
        response = requests.get(settings.overpass_endpoint.rstrip('/') + '/status', headers=get_http_headers())
        status = response.text.split('\n')[3]
        status_first_token = status.split(' ')[0]
    except Exception:
        # if we cannot reach the status endpoint or parse its output, log an
        # error and return default duration
        log('Unable to query {}/status'.format(settings.overpass_endpoint.rstrip('/')), level=lg.ERROR)
        return default_duration

    try:
        # if first token is numeric, it's how many slots you have available - no
        # wait required
        available_slots = int(status_first_token)
        pause_duration = 0
    except Exception:
        # if first token is 'Slot', it tells you when your slot will be free
        if status_first_token == 'Slot':
            utc_time_str = status.split(' ')[3]
            utc_time = date_parser.parse(utc_time_str).replace(tzinfo=None)
            pause_duration = math.ceil((utc_time - dt.datetime.utcnow()).total_seconds())
            pause_duration = max(pause_duration, 1)

        # if first token is 'Currently', it is currently running a query so
        # check back in recursive_delay seconds
        elif status_first_token == 'Currently':
            time.sleep(recursive_delay)
            pause_duration = get_pause_duration()

        else:
            # any other status is unrecognized - log an error and return default
            # duration
            log('Unrecognized server status: "{}"'.format(status), level=lg.ERROR)
            return default_duration

    return pause_duration


def osm_polygon_download(query, limit=1, polygon_geojson=1):

    # define the parameters
    params = OrderedDict()
    params['format'] = 'json'
    params['limit'] = limit
    params['dedupe'] = 0  # prevent OSM from deduping results so we get precisely 'limit' # of results
    params['polygon_geojson'] = polygon_geojson

    # add the structured query dict (if provided) to params, otherwise query
    # with place name string
    if isinstance(query, str):
        params['q'] = query
    elif isinstance(query, dict):
        # add the query keys in alphabetical order so the URL is the same string
        # each time, for caching purposes
        for key in sorted(list(query.keys())):
            params[key] = query[key]
    else:
        raise TypeError('query must be a dict or a string')

    # request the URL, return the JSON
    response_json = nominatim_request(params=params, timeout=30)
    return response_json


def nominatim_request(params, type="search", pause_duration=1, timeout=30, error_pause_duration=180):

    known_requests = {"search", "reverse", "lookup"}
    if type not in known_requests:
        raise ValueError(
            "The type of Nominatim request is invalid. Please choose one of {{'search', 'reverse', 'lookup'}}")

    # prepare the Nominatim API URL and see if request already exists in the
    # cache
    url = settings.nominatim_endpoint.rstrip('/') + '/{}'.format(type)
    prepared_url = requests.Request('GET', url, params=params).prepare().url
    cached_response_json = get_from_cache(prepared_url)

    if settings.nominatim_key:
        params['key'] = settings.nominatim_key

    if cached_response_json is not None:
        # found this request in the cache, just return it instead of making a
        # new HTTP call
        return cached_response_json

    else:
        # if this URL is not already in the cache, pause, then request it
        log('Pausing {:,.2f} seconds before making API GET request'.format(pause_duration))
        time.sleep(pause_duration)
        start_time = time.time()
        log('Requesting {} with timeout={}'.format(prepared_url, timeout))
        response = requests.get(url, params=params, timeout=timeout, headers=get_http_headers())

        # get the response size and the domain, log result
        size_kb = len(response.content) / 1000.
        domain = re.findall(r'(?s)//(.*?)/', url)[0]
        log('Downloaded {:,.1f}KB from {} in {:,.2f} seconds'.format(size_kb, domain, time.time() - start_time))

        try:
            response_json = response.json()
            save_to_cache(prepared_url, response_json)
        except Exception:
            # 429 is 'too many requests' and 504 is 'gateway timeout' from server
            # overload - handle these errors by recursively calling
            # nominatim_request until we get a valid response
            if response.status_code in [429, 504]:
                # pause for error_pause_duration seconds before re-trying request
                log(
                    'Server at {} returned status code {} and no JSON data. Re-trying request in {:.2f} seconds.'.format(
                        domain,
                        response.status_code,
                        error_pause_duration),
                    level=lg.WARNING)
                time.sleep(error_pause_duration)
                response_json = nominatim_request(params=params, pause_duration=pause_duration, timeout=timeout)

            # else, this was an unhandled status_code, throw an exception
            else:
                log('Server at {} returned status code {} and no JSON data'.format(domain, response.status_code),
                    level=lg.ERROR)
                raise Exception(
                    'Server returned no JSON data.\n{} {}\n{}'.format(response, response.reason, response.text))

        return response_json


def overpass_request(data, pause_duration=None, timeout=180, error_pause_duration=None):

    # define the Overpass API URL, then construct a GET-style URL as a string to
    # hash to look up/save to cache
    url = settings.overpass_endpoint.rstrip('/') + '/interpreter'
    prepared_url = requests.Request('GET', url, params=data).prepare().url
    cached_response_json = get_from_cache(prepared_url)

    if cached_response_json is not None:
        # found this request in the cache, just return it instead of making a
        # new HTTP call
        return cached_response_json

    else:
        # if this URL is not already in the cache, pause, then request it
        if pause_duration is None:
            this_pause_duration = get_pause_duration()
        log('Pausing {:,.2f} seconds before making API POST request'.format(this_pause_duration))
        time.sleep(this_pause_duration)
        start_time = time.time()
        log('Posting to {} with timeout={}, "{}"'.format(url, timeout, data))
        response = requests.post(url, data=data, timeout=timeout, headers=get_http_headers())

        # get the response size and the domain, log result
        size_kb = len(response.content) / 1000.
        domain = re.findall(r'(?s)//(.*?)/', url)[0]
        log('Downloaded {:,.1f}KB from {} in {:,.2f} seconds'.format(size_kb, domain, time.time() - start_time))

        try:
            response_json = response.json()
            if 'remark' in response_json:
                log('Server remark: "{}"'.format(response_json['remark'], level=lg.WARNING))
            save_to_cache(prepared_url, response_json)
        except Exception:
            # 429 is 'too many requests' and 504 is 'gateway timeout' from server
            # overload - handle these errors by recursively calling
            # overpass_request until we get a valid response
            if response.status_code in [429, 504]:
                # pause for error_pause_duration seconds before re-trying request
                if error_pause_duration is None:
                    error_pause_duration = get_pause_duration()
                log(
                    'Server at {} returned status code {} and no JSON data. Re-trying request in {:.2f} seconds.'.format(
                        domain,
                        response.status_code,
                        error_pause_duration),
                    level=lg.WARNING)
                time.sleep(error_pause_duration)
                response_json = overpass_request(data=data, pause_duration=pause_duration, timeout=timeout)

            # else, this was an unhandled status_code, throw an exception
            else:
                log('Server at {} returned status code {} and no JSON data'.format(domain, response.status_code),
                    level=lg.ERROR)
                raise Exception(
                    'Server returned no JSON data.\n{} {}\n{}'.format(response, response.reason, response.text))

        return response_json
