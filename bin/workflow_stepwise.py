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


def get_masst_related_data(
    smiles: str,
    sqlite_path: str = None,
    api_endpoint: str = "http://127.0.0.1:8001/masst_records",
    timeout: int = 10,
    chunk_size: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given a SMILES, returns (masst_df, library_df, redu_df).
    If sqlite_path exists, queries local SQLite; otherwise uses the Datasette CSV API.
    Splits large IN(...) queries into chunks of size `chunk_size`.
    """

    # 1) SMILES → InChIKey prefix
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    prefix = inchi.MolToInchiKey(mol).split('-')[0]

    # ——— helpers ———
    def fetch_csv(sql: str) -> pd.DataFrame:
        print(f"[API ] Querying with SQL: {sql}")
        resp = requests.get(
            f"{api_endpoint}.csv",
            params={"sql": sql, "_stream": "on"},
            timeout=timeout
        )
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        print(f"[API ] returned {len(df)} rows")
        return df

    def fetch_sqlite(sql: str) -> pd.DataFrame:
        print(f"[SQL ] Querying with SQL: {sql}")
        with sqlite3.connect(sqlite_path) as conn:
            df = pd.read_sql(sql, conn)
        print(f"[SQL ] returned {len(df)} rows")
        return df

    use_sqlite = bool(sqlite_path and os.path.isfile(sqlite_path))
    fetch = fetch_sqlite if use_sqlite else fetch_csv

    def batched_fetch(template_sql: str, id_list: list[int]) -> pd.DataFrame:
        """
        Runs template_sql multiple times, substituting `{ids}` with comma–sep chunks.
        Returns the concatenated DataFrame.
        """
        if not id_list:
            print("[BATCH] No IDs to fetch.")
            return pd.DataFrame()
        dfs = []
        for i in range(0, len(id_list), chunk_size):
            chunk = id_list[i : i + chunk_size]
            sql = template_sql.format(ids=",".join(map(str, chunk)))
            print(f"[BATCH] chunk {i//chunk_size+1}: {len(chunk)} IDs")
            df = fetch(sql)
            if not df.empty:
                dfs.append(df)
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            print(f"[BATCH] total returned {len(result)} rows")
            return result
        else:
            print("[BATCH] returned 0 rows")
            return pd.DataFrame()

    # 2) library_table
    lib_sql = (
        "SELECT * FROM library_table "
        f"WHERE InChIKey_smiles_firstBlock = '{prefix}'"
    )
    print(f"[STEP 1] library_table for prefix='{prefix}'")
    library_df = fetch(lib_sql)
    if library_df.empty:
        print("[STEP 1] no library hits → exiting")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 3) masst_table (batch on spectrum_id_int)
    sids = library_df['spectrum_id_int'].dropna().unique().tolist()
    print(f"[STEP 2] masst_table for {len(sids)} spectrum_id_ints")
    if not sids:
        return pd.DataFrame(), library_df, pd.DataFrame()
    masst_sql_tmpl = "SELECT * FROM masst_table WHERE spectrum_id_int IN ({ids})"
    masst_df = batched_fetch(masst_sql_tmpl, sids)
    if masst_df.empty:
        print("[STEP 2] no masst hits → exiting")
        return pd.DataFrame(), library_df, pd.DataFrame()

    # 3a) Join in actual mri strings from mri_table
    mids = masst_df['mri_id_int'].dropna().unique().tolist()
    if mids:
        print(f"[STEP 3a] fetching mri strings for {len(mids)} mri_id_ints")
        mri_sql_tmpl = "SELECT mri_id_int, mri FROM mri_table WHERE mri_id_int IN ({ids})"
        mri_map_df = batched_fetch(mri_sql_tmpl, mids)
        if not mri_map_df.empty:
            masst_df = masst_df.merge(mri_map_df, on='mri_id_int', how='left')
            print(f"[STEP 3a] merged mri strings into masst_df")
        else:
            masst_df['mri'] = None
    else:
        masst_df['mri'] = None

    # 3b) Join in actual spectrum_id from library_table
    print(f"[STEP 3b] merging spectrum_id for {len(sids)} spectrum_id_ints")
    spec_map_df = library_df[['spectrum_id_int', 'spectrum_id']].drop_duplicates()
    masst_df = masst_df.merge(spec_map_df, on='spectrum_id_int', how='left')
    print(f"[STEP 3b] merged spectrum_id into masst_df")

    # 4) redu_table (direct on mri_id_int, batch)
    print(f"[STEP 4] redu_table for {len(mids)} mri_id_ints")
    if not mids:
        print("[STEP 4] no mri_ids → returning masst+library only")
        return masst_df, library_df, pd.DataFrame()
    redu_sql_tmpl = "SELECT * FROM redu_table WHERE mri_id_int IN ({ids})"
    redu_df = batched_fetch(redu_sql_tmpl, mids)

    return masst_df, library_df, redu_df



def GetLibraryConflicts(df_libfasst, df_library_adduct_inchikey_smiles):

    print("Preparing input tables for library conflicts...")
    df_library_adduct_inchikey_smiles = df_library_adduct_inchikey_smiles.copy()
    df_libfasst = df_libfasst.copy()

    df_library_adduct_inchikey_smiles.rename(columns={'spectrum_id': 'query_spectrum_id'}, inplace=True)
    df_library_adduct_inchikey_smiles = df_library_adduct_inchikey_smiles.drop_duplicates(subset=['query_spectrum_id'])

    df_libfasst = pd.merge(df_libfasst, df_library_adduct_inchikey_smiles, on="query_spectrum_id", how="left")
    df_libfasst.rename(columns={
        'Adduct': 'query_adduct',
        'InChIKey_smiles': 'query_inchikey_smiles',
        'Smiles': 'query_smiles'
    }, inplace=True)

    df_library_adduct_inchikey_smiles.rename(columns={'query_spectrum_id': 'GNPSLibraryAccession'}, inplace=True)
    df_libfasst = pd.merge(df_libfasst, df_library_adduct_inchikey_smiles, on="GNPSLibraryAccession", how="left")
    df_libfasst.rename(columns={
        'Adduct': 'matching_adduct',
        'InChIKey_smiles': 'matching_inchikey_smiles',
        'Smiles': 'matching_smiles',
        'GNPSLibraryAccession': 'matching_spectrum_id'
    }, inplace=True)

    df_libfasst = df_libfasst[df_libfasst['matching_inchikey_smiles'].notnull()]

    print("Counting same and different InChIKey matches...")
    same_inchikey = df_libfasst[df_libfasst['matching_inchikey_smiles'] == df_libfasst['query_inchikey_smiles']]
    same_counts = same_inchikey.groupby('query_spectrum_id').size().reset_index(name='same_molecule_spectral_match_count')

    different_inchikey = df_libfasst[df_libfasst['matching_inchikey_smiles'] != df_libfasst['query_inchikey_smiles']]
    diff_row_counts = (
        different_inchikey.groupby('query_spectrum_id')
        .size()
        .reset_index(name='different_molecule_spectral_match_count')
    )
    diff_unique_counts = (
        different_inchikey.groupby('query_spectrum_id')['matching_inchikey_smiles']
        .nunique()
        .reset_index(name='different_molecule_count')
    )

    print("Extracting static query info...")
    query_info = df_libfasst[['query_spectrum_id', 'query_adduct', 'query_smiles', 'query_inchikey_smiles']].drop_duplicates()



    conflict_map = defaultdict(dict)

    for query_id, group in df_libfasst.groupby('query_spectrum_id'):
        q_inchikey = group['query_inchikey_smiles'].iloc[0]
        
        for _, row in group.iterrows():
            m_inchikey = row['matching_inchikey_smiles']
            m_smiles = row['matching_smiles']
            
            if pd.isna(m_inchikey) or pd.isna(m_smiles):
                continue
            if m_inchikey == q_inchikey:
                continue
            
            # store one representative SMILES per InChIKey
            # After — store a tuple of (SMILES, adduct)
            if m_inchikey not in conflict_map[query_id]:
                conflict_map[query_id][m_inchikey] = (m_smiles, row['matching_adduct'])


    # Identify all unique conflicting InChIKeys across all queries
    all_conflicting_inchikeys = sorted({ik for row in conflict_map.values() for ik in row})

    # Now pivot the conflict_map to a DataFrame with stable column order
    rows = []
    for query_id, inchikey_to_smiles in conflict_map.items():
        row = {"query_spectrum_id": query_id}
        for idx, inchikey in enumerate(all_conflicting_inchikeys):
            val = inchikey_to_smiles.get(inchikey)
            if val:
                smi, adduct = val
                row[f"conflicting_molecule_{idx+1}"] = smi
                row[f"conflicting_molecule_{idx+1}_adduct"] = adduct
        rows.append(row)

    # Build the DataFrame
    smiles_wide = pd.DataFrame(rows)

    # Fill missing conflicts with None
    smiles_wide = smiles_wide.fillna(value=pd.NA)


    print("Merging all conflict information into summary table...")
    summary = query_info.merge(same_counts, on='query_spectrum_id', how='left')
    summary = summary.merge(diff_row_counts, on='query_spectrum_id', how='left')
    summary = summary.merge(diff_unique_counts, on='query_spectrum_id', how='left')
    if not smiles_wide.empty:
        summary = summary.merge(smiles_wide, on='query_spectrum_id', how='left')

    summary[['same_molecule_spectral_match_count', 'different_molecule_spectral_match_count', 'different_molecule_count']] = (
        summary[['same_molecule_spectral_match_count', 'different_molecule_spectral_match_count', 'different_molecule_count']]
        .fillna(0).astype(int)
    )

    summary = summary.sort_values(by='different_molecule_spectral_match_count', ascending=False)


    print("Saving library conflicts summary to output/structureMASST_library_conflicts.tsv")

    return summary

def retrieveSpectraCandidates(df_library, input_structure, get_conflicts = False, analog=False, database = 'metabolomicspanrepo_index_nightly', precursor_mz_tol = 0.05, fragment_mz_tol = 0.05, min_cos = 0.7, matching_peaks = 6, cache = "Yes"):

    print("Filtering library for valid SMILES and InChIKey_smiles...")
    df_library = df_library[
        df_library['Smiles'].notnull() & (df_library['Smiles'] != '') & (df_library['Smiles'] != 'NaN') &
        df_library['InChIKey_smiles'].notnull() & (df_library['InChIKey_smiles'] != '') & (df_library['InChIKey_smiles'] != 'NaN') &
        (df_library['ppmBetweenExpAndThMass'] <= 20) &
        ~df_library['msMassAnalyzer'].isin(['quadrupole', 'ion trap'])
    ]
    
    df_library_adduct_inchikey_smiles = df_library[['spectrum_id', 'Adduct', 'InChIKey_smiles', 'Smiles']]
    print("Matching input structure to library...")
    structure_type = detect_smiles_or_smarts(input_structure)
    df_library_structurematch = fetch_and_match_smiles(df_library, input_structure, smiles_type = structure_type, max_by_grp = 1000, max_overall = 1000)
    df_library_structurematch = df_library_structurematch[["Compound_Name", "query_spectrum_id", "inchikey_first_block", 'Adduct', 'collision_energy', 'msMassAnalyzer', 'Ion_Mode']]


    libfasst_results = []
    df_libfasst = pd.DataFrame()
    if len(df_library_structurematch) <= 20 and get_conflicts:
        for index, row in df_library_structurematch.iterrows():
            print(f"Querying FASST for spectrum {row['query_spectrum_id']} in {database} and GNPS library...")
            lib_id = row['query_spectrum_id']
            df_libresponse = query_fasst_usi(lib_id, 'gnpslibrary', analog=analog, precursor_mz_tol=precursor_mz_tol,
                                            fragment_mz_tol=fragment_mz_tol, min_cos=min_cos, matching_peaks=matching_peaks,
                                            cache=cache)
            
            if not df_libresponse.empty:
                # eliminate GNPSLibraryAccession not present in spectrum_id
                df_libresponse = df_libresponse[df_libresponse['GNPSLibraryAccession'].isin(df_library['spectrum_id'])]

            libfasst_results.append(pd.DataFrame())
            libfasst_results.append(df_libresponse)

        print("Combining FASST results...")
        df_libfasst = pd.concat(libfasst_results)

    if not df_libfasst.empty and get_conflicts:
        df_libfasst = df_libfasst[['GNPSLibraryAccession', 'query_spectrum_id']]

        print("Analyzing library conflicts...")
        df_library_conflicts = GetLibraryConflicts(df_libfasst, df_library_adduct_inchikey_smiles)

        # Add columns "same_molecule_spectral_match_count", "different_molecule_spectral_match_count", "different_molecule_count" to df_library_structurematch
        df_library_structurematch = df_library_structurematch.merge(
            df_library_conflicts[['query_spectrum_id', 'same_molecule_spectral_match_count', 
                                'different_molecule_spectral_match_count', 'different_molecule_count']],
            on='query_spectrum_id', how='left'
        )

        # Where no values were added, fill with 0. if nothing was added because
        df_library_structurematch.fillna(0, inplace=True)

    else:        

        df_library_conflicts = pd.DataFrame(columns=[
            'query_spectrum_id', 'same_molecule_spectral_match_count', 'different_molecule_spectral_match_count', 'different_molecule_count'
        ])

    df_library_structurematch['Smiles'] = input_structure


    return df_library_conflicts, df_library_structurematch

    
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
    cache: str = "Yes"
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
    sql = """
    SELECT * FROM redu_table
    """

    url = "https://masst-records.gnps2.org/masst_records_copy.json"

    params = {
        "sql": sql,
        "_shape": "objects",
        "_size": 1000,  
    }

    all_rows = []
    while url:
        print(f"Fetching: {url}")
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        all_rows.extend(data["rows"])
        next_url = data.get("next_url")
        url = f"https://masst-records.gnps2.org{next_url}" if next_url else None
        params = {}  

    redu_df = pd.DataFrame(all_rows)

    columns_to_exclude = ['filename', 'TermsofPosition', 'ComorbidityListDOIDIndex', 'SampleCollectionDateandTime', 'ENVOBroadScale', 'ENVOLocalScale', 'ENVOMediumScale', 'qiita_sample_name',
                          'UniqueSubjectID', 'UBERONOntologyIndex', 'DOIDOntologyIndex', 'ENVOEnvironmentBiomeIndex', 'ENVOEnvironmentMaterialIndex', 'ENVOLocalScaleIndex', 'ENVOBroadScaleIndex',
                          'ENVOMediumScaleIndex', 'classification', 'MS2spectra_count']

    redu_df = redu_df.drop(columns=columns_to_exclude, errors='ignore')

    # 1. Run FASST queries and collect non-empty responses
    responses = []
    for spectrum_id in library_subset['query_spectrum_id']:
        print(f"Querying FASST for spectrum {spectrum_id} in {database}...")
        df = query_fasst_usi(
            spectrum_id,
            database,
            analog=analog,
            precursor_mz_tol=precursor_mz_tol,
            fragment_mz_tol=fragment_mz_tol,
            min_cos=min_cos,
            matching_peaks=matching_peaks,
            cache=cache,
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

    # in every row add USI + :scan: + scan_id (as str)
    redu_enriched["USI"] = redu_enriched["USI"] + ":scan:" + redu_enriched["scan_id"].astype(str)

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

