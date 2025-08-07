import pandas as pd
import argparse
import os
import json
import time
import requests
from tqdm import tqdm
import sys

HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'GNPSDataPackage'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from gnpsdata.fasst import query_fasst_api_usi



def query_fasst_usi(usi, database='metabolomicspanrepo_index_nightly',
                    analog=False, precursor_mz_tol=0.05,
                    fragment_mz_tol=0.05, min_cos=0.7, matching_peaks=6,
                    cache="Yes", modimass=None, elimination=False, addition=False):

   
    if usi.startswith("mzspec"):
        usi_full = usi
    else:
        usi_full = make_library_usi(usi)
        

    params = {
        "usi": usi_full,
        "library": database,
        "analog": "Yes" if analog else "No",
        "pm_tolerance": precursor_mz_tol,
        "fragment_tolerance": fragment_mz_tol,
        "cosine_threshold": min_cos,
        "cache": cache
    }
    
    try:
        modimass_val = float(modimass)
    except (TypeError, ValueError):
        modimass_val = None
    
    print(f"Received modimass: {modimass_val}")
    if analog and modimass_val is not None:
        params['delta_mass_below'] = modimass_val + 1
        params['delta_mass_above'] = modimass_val + 1
    elif analog:
        params['delta_mass_below'] = 100
        params['delta_mass_above'] = 100

    for attempt in range(3):
        try:

            print("Query parameters:")
            for key, val in params.items():
                print(f"  {key}: {val}")

            r = query_fasst_api_usi(params['usi'], params['library'], host="https://api.fasst.gnps2.org",
                                    analog=analog, precursor_mz_tol=params['pm_tolerance'],
                                    fragment_mz_tol=params['fragment_tolerance'], min_cos=params['cosine_threshold'],
                                    lower_delta=params.get('delta_mass_below', 100),
                                    upper_delta=params.get('delta_mass_above', 100),
                                    cache=params['cache'])

            response_list = r['results']
            
            if len(response_list) > 0:
                df = pd.DataFrame(response_list)

                print(f"Response from FASST: {len(df)} results found for USI {usi_full}")

                if analog == False:
                    df = df[df['Delta Mass'].abs() <= precursor_mz_tol]
                    print(f"Delta Mass filter applied: {precursor_mz_tol}")

                elif analog == True:
                    df = df[(df['Delta Mass'].abs() >= 5) | (df['Delta Mass'].abs() <= precursor_mz_tol)]
                    df.loc[df['Delta Mass'].abs() <= precursor_mz_tol, 'Modified'] = 'no'
                    df.loc[df['Delta Mass'] > precursor_mz_tol, 'Modified'] = 'addition'
                    df.loc[df['Delta Mass'] < -precursor_mz_tol, 'Modified'] = 'elimination'


                    if modimass_val is not None:
                        df = df[
                            (df['Delta Mass'].abs() <= precursor_mz_tol) |
                            ((df['Delta Mass'].abs() - modimass_val).abs() <= precursor_mz_tol)
                        ]

                    if elimination and not addition:
                        df = df[(df['Delta Mass'].abs() <= precursor_mz_tol) | (df['Delta Mass'] < 0)]

                    if addition and not elimination:
                        df = df[(df['Delta Mass'].abs() <= precursor_mz_tol) | (df['Delta Mass'] > 0)]

                    

                df = df[df['Matching Peaks'] >= matching_peaks]
                df['query_spectrum_id'] = usi
                df.drop(columns=[
                    'Charge', 'Unit Delta Mass', 'Status', 'Query Filename', 
                    'Index UnitPM', 'Index IdxInUnitPM', 'Filtered Input Spectrum Path', 
                    'Query Scan'
                ], inplace=True)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                return pd.DataFrame()


def make_library_usi(lib_id):
    if lib_id.startswith("CCMSLIB"):
        return "mzspec:GNPS:GNPS-LIBRARY:accession:{}".format(lib_id)
    else:
        return "mzspec:MASSBANK::accession:{}".format(lib_id)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query FASST USI')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('--database', help='Database to query', default='metabolomicspanrepo_index_latest')
    parser.add_argument('--analog', help='Analog search', default=False, type=bool)
    parser.add_argument('--precursor_mz_tol', help='Precursor m/z tolerance', default=0.05, type=float)
    parser.add_argument('--fragment_mz_tol', help='Fragment m/z tolerance', default=0.05, type=float)
    parser.add_argument('--min_cos', help='Minimum cosine score', default=0.7, type=float)
    parser.add_argument('--cache', help='Use cache', default="Yes")
    parser.add_argument('--test', help='test', default=False, type=bool)
    args = parser.parse_args()


    if args.test:

        lib_id = args.input

        usi = make_library_usi(lib_id)
        print(usi)
        df_response = query_fasst_usi(usi, args.database, analog=args.analog, precursor_mz_tol=args.precursor_mz_tol,
                                      fragment_mz_tol=args.fragment_mz_tol, min_cos=args.min_cos, cache=args.cache)
        


        print(f"{lib_id} returns {len(df_response)} results")

        print(f"Returned columns are: {df_response.columns}")

        print(df_response.head())

        print("Testing masst_records")
