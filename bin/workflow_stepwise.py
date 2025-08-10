import pandas as pd 
from bin.match_smiles import fetch_and_match_smiles, detect_smiles_or_smarts
from bin.match_smiles import detect_smiles_or_smarts
from bin.run_fasst import query_fasst_usi
# from match_smiles import fetch_and_match_smiles, detect_smiles_or_smarts
# from run_fasst import query_fasst_usi
# from make_linkouts import create_gnps_link
import argparse
from collections import defaultdict
import os
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from io import StringIO
import sqlite3
from urllib.parse import quote_plus
import sys
from bin.run_masstRecords_queries import _get_fetcher


HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'GNPSDataPackage'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)


from gnpsdata import fasst


FASST_API_SERVER_URL = "https://api.fasst.gnps2.org"


def make_library_usi(lib_id):
    if lib_id.startswith("CCMSLIB"):
        return "mzspec:GNPS:GNPS-LIBRARY:accession:{}".format(lib_id)
    else:
        return "mzspec:MASSBANK::accession:{}".format(lib_id)
   
def retrieve_raw_data_matches(
    library_subset: pd.DataFrame,
    analog: bool = False,
    elimination: bool = False,
    addition: bool = False,
    modimass: float | None = None,
    modification_condition: str = None,
    database: str = 'metabolomicspanrepo_index_nightly',
    precursor_mz_tol: float = 0.05,
    fragment_mz_tol: float = 0.05,
    min_cos: float = 0.7,
    matching_peaks: int = 6,
    cache: str = "Yes",
    sqlite_path: str | None = None,
    api_endpoint: str = "http://127.0.0.1:8001/masst_records",
    timeout: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Query FASST for each spectrum in library_subset and optionally merge ReDU metadata.

    Args:
        library_subset: DataFrame with a 'query_spectrum_id' column.
        analog: Whether to run an analog search.
        database: FASST database name.
        precursor_mz_tol: Precursor m/z tolerance.
        fragment_mz_tol: Fragment m/z tolerance.
        min_cos: Minimum cosine score.
        matching_peaks: Minimum number of matching peaks.
        cache: Cache policy.

    Returns:
        raw_matches: concatenated FASST responses with 'spectrum_id' column.
        redu_enriched: raw_matches merged with redu_df (empty if redu_df is None/empty).
    """


    # 0. load redu data
    print("Loading ReDU table...")

    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)

    # get the column names
    redu_columns = fetch("SELECT name FROM pragma_table_info('redu_table')")
    redu_columns_list = redu_columns["name"].tolist()

    # exclude unwanted columns
    columns_to_exclude = [
        "filename","TermsofPosition","ComorbidityListDOIDIndex","SampleCollectionDateandTime",
        "ENVOBroadScale","ENVOLocalScale","ENVOMediumScale","qiita_sample_name","UniqueSubjectID",
        "UBERONOntologyIndex","DOIDOntologyIndex","ENVOEnvironmentBiomeIndex",
        "ENVOEnvironmentMaterialIndex","ENVOLocalScaleIndex","ENVOBroadScaleIndex",
        "ENVOMediumScaleIndex","classification","MS2spectra_count"
    ]
    cols = [c for c in redu_columns_list if c not in columns_to_exclude]
    col_sql = ", ".join([f'"{c}"' for c in cols])

    # count total rows
    total_rows = int(fetch("SELECT COUNT(*) as n FROM redu_table")["n"].iloc[0])

    # function to fetch one page
    def fetch_page(offset, limit):
        sql = f"SELECT {col_sql} FROM redu_table LIMIT {limit} OFFSET {offset}"
        return fetch(sql)

    # loop in batches
    chunk_size = 50000  # adjust as needed
    dfs = []
    for offset in range(0, total_rows, chunk_size):
        print(f"[PAGE] offset {offset} / {total_rows}")
        df_chunk = fetch_page(offset, chunk_size)
        dfs.append(df_chunk)

    # combine
    redu_df = pd.concat(dfs, ignore_index=True)
    print(f"Final total: {len(redu_df)} rows")

    print("ReDU DataFrame loaded with shape:", redu_df.shape)

    # 1. Run FASST queries and collect non-empty responses
    status_results_list = []
    for spectrum_id in library_subset['query_spectrum_id']:
        usi_full = make_library_usi(spectrum_id)
        print("submitted", usi_full)
        results = fasst.query_fasst_api_usi(usi_full, database, host=FASST_API_SERVER_URL, analog=analog, 
                                            lower_delta=170, upper_delta=170, precursor_mz_tol=precursor_mz_tol, fragment_mz_tol=fragment_mz_tol, 
                                            min_cos=min_cos, cache=cache, blocking=False)
        
        status_results_list.append(results)
        
        
    responses = []
    for status in status_results_list:
        print(f"Checking status for {status}")
        df = query_fasst_usi(
            status,
            spectrum_id,
            precursor_mz_tol=precursor_mz_tol,
            analog=analog,
            matching_peaks=matching_peaks,
            modimass=modimass,
            elimination=elimination,
            addition=addition
        )
        if not df.empty:
            responses.append(df)

    # 2. Combine all responses
    if not responses:
        print("No raw data matches found.")
        return pd.DataFrame(), pd.DataFrame()
    raw_matches = pd.concat(responses, ignore_index=True)
    raw_matches.rename(columns={'GNPSLibraryAccession': 'spectrum_id'}, inplace=True)

    # 3. If ReDU data provided, merge and return enriched DataFrame
    if redu_df is None or redu_df.empty:
        return raw_matches, pd.DataFrame()

    redu_enriched = add_redu(raw_matches, redu_df, modification_condition=modification_condition)
    
    # add Smiles column from library_subset to redu_enriched
    if 'Smiles' in library_subset.columns:
        redu_enriched = redu_enriched.merge(
            library_subset[['query_spectrum_id', 'Smiles', 'Adduct', 'Compound_Name']],
            left_on='query_spectrum_id',
            right_on='query_spectrum_id',
            how='left'
        )
        redu_enriched.rename(columns={'Smiles': 'query_smiles'}, inplace=True)


    # make library usis for the links
    redu_enriched["lib_usi"] = redu_enriched["query_spectrum_id"].apply(
        lambda x: (
            f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}" if x.startswith("CCMSLIB")
            else f"mzspec:MASSBANK::accession:{x}" 
        )
    )

    if 'Modified' in redu_enriched.columns:
        def build_modifinder_link(row):
            usi1 = quote_plus(f"{row['USI']}")
            usi2 = quote_plus(row['lib_usi'])
            return (
                f"https://modifinder.gnps2.org/"
                f"?USI1={usi2}"
                f"&USI2={usi1}"
                f"&SMILES1={quote_plus(row['query_smiles'])}"
                f"&SMILES2&Helpers=&adduct={quote_plus(row['Adduct'])}"
                "&ppm_tolerance=25&filter_peaks_variable=0.01"
            )
               

        redu_enriched["modification_site"] = redu_enriched.apply(
            lambda row: build_modifinder_link(row)
            if (row["Modified"] != "no" and row["Adduct"] in 
                ['[M+H]1+', '[M-H]1', '[M+Na]1+', '[M+NH4]1+', '[M+K]1+', '[M+Cl]1-', '[M+Br]1-'])
            else '',
            axis=1
        )
    
    return raw_matches, redu_enriched
    

def add_redu(
    raw_matches: pd.DataFrame,
    redu_df: pd.DataFrame,
    modification_condition: str = None
) -> pd.DataFrame:
    """
    Enrich raw_matches with ReDU metadata from redu_df via the 'mri' key.

    Steps:
    1. Return early if no ReDU data is provided.
    2. Make a local copy of raw_matches and sort by descending Cosine and Matching Peaks.
    3. If 'USI' exists, split it into 'mri' and 'scan_id' on ':scan:'.
    4. Deduplicate on 'mri', keeping the highest-scoring match.
    5. Rename redu_df.USIs to 'mri' if necessary.
    6. Inner-merge on 'mri' to retain only matches with ReDU metadata.
    7. Fill any NaNs in ReDU columns (those starting with 'redu_') with 'unknown'.
    """
    if redu_df.empty:
        print("[add_redu] No ReDU data provided; returning original matches.")
        return raw_matches.copy()

    # 1. Prepare and sort raw matches
    df = raw_matches.copy()
    
    # 2. Extract 'mri' and 'scan_id' from USI if present
    if "USI" in df.columns:
        df[["mri", "scan_id"]] = df["USI"].str.split(":scan:", n=1, expand=True)

    df = df.sort_values(
        by=["Cosine", "Matching Peaks"], 
        ascending=[False, False]
    )

    if 'Modified' in df.columns:
        unique_by_columns = ['mri', 'Delta Mass']
    else:
        unique_by_columns = ['mri']

    # 3. Keep only the top match per 'mri'
    if "mri" in df.columns:
        df = df.drop_duplicates(subset=unique_by_columns, keep="first")
    else:
        print("[add_redu] Warning: 'mri' column not found in matches; merging may fail.")



    # 4. Prepare redu_df for merging
    df_redu = redu_df.copy()
    if "USI" in df_redu.columns and "mri" not in df_redu.columns:
        df_redu = df_redu.rename(columns={"USI": "mri"})

    # 5. Merge matches with ReDU metadata
    merged = df.merge(df_redu, on="mri", how="inner")
    print(f"[add_redu] Merged {len(df_redu)} ReDU records; result has {len(merged)} rows.")

    # 6. Fill missing values in any ReDU-specific columns
    redu_cols = [col for col in merged.columns if col.startswith("redu_")]
    if redu_cols:
        merged[redu_cols] = merged[redu_cols].fillna("unknown")


    if 'Modified' in merged.columns and modification_condition:
        if modification_condition == "Raw file":
            modification_condition = 'mri'

        valid_groups = merged[merged['Modified'] == 'no'][modification_condition].unique()
        merged = merged[merged[modification_condition].isin(valid_groups)]




        # drop columns used for link generation
        #merged = merged.drop(columns=["lib_usi"], errors="ignore")
        # merged = merged.drop(columns=["query_smiles"], errors="ignore")

    return merged

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Query FASST USI')
    parser.add_argument('input', help='Library csv path')
    parser.add_argument('structure', help='structure')
    parser.add_argument('--database', help='Database to query', default='metabolomicspanrepo_index_nightly')
    parser.add_argument('--analog', help='Analog search', default=False, type=bool)
    parser.add_argument('--precursor_mz_tol', help='Precursor m/z tolerance', default=0.05, type=float)
    parser.add_argument('--fragment_mz_tol', help='Fragment m/z tolerance', default=0.05, type=float)
    parser.add_argument('--min_cos', help='Minimum cosine score', default=0.7, type=float)
    parser.add_argument('--cache', help='Use cache', default="Yes")
    parser.add_argument('--test', help='test', default=False, type=bool)
    args = parser.parse_args()

    if args.test:
        print("Test mode enabled. Using hardcoded input structure.")
        input_structure = 'C[C@H](CCC(N[C@H](C(O)=O)CC1=CNC2=C1C=CC=C2)=O)[C@H]3CC[C@@]4([H])[C@]5([H])[C@H](O)C[C@]6([H])C[C@H](O)CC[C@]6(C)[C@H]5C[C@H](O)[C@@]43C' 
    else:
        input_structure = args.structure

    print("Starting structureMASST workflow...")
    # df, df_library_conflicts = structureMASST(library=args.input, input_structure=input_structure, analog=args.analog, database=args.database,
    #                     precursor_mz_tol=args.precursor_mz_tol, fragment_mz_tol=args.fragment_mz_tol, min_cos=args.min_cos, 
    #                     cache=args.cache)

    _, df_library_structurematch = retrieveSpectraCandidates(args.input, input_structure)

    matches = retrieve_raw_data_matches(
        df_library_structurematch, analog=args.analog, database=args.database,
        precursor_mz_tol=args.precursor_mz_tol, fragment_mz_tol=args.fragment_mz_tol,
        min_cos=args.min_cos, matching_peaks=6, cache=args.cache
    )

    print("Saving structureMASST results to tsv")
    matches.to_csv('output/df_raw_matches.tsv', sep="\t", index=False)

