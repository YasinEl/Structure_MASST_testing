import pandas as pd 
from bin.match_smiles import fetch_and_match_smiles
from bin.match_smiles import detect_smiles_or_smarts
from bin.run_fasst import query_fasst_usi
# from match_smiles import fetch_and_match_smiles
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


# ——— Shared helpers ———

def _fetch_csv(sql: str, api_endpoint: str, timeout: int) -> pd.DataFrame:
    print(f"[API ] Querying with SQL: {sql}")
    resp = requests.get(f"{api_endpoint}.csv", params={"sql": sql, "_stream": "on"}, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    print(f"[API ] returned {len(df)} rows")
    return df

def _fetch_sqlite(sql: str, sqlite_path: str) -> pd.DataFrame:
    print(f"[SQL ] Querying with SQL: {sql}")
    with sqlite3.connect(sqlite_path) as conn:
        df = pd.read_sql(sql, conn)
    print(f"[SQL ] returned {len(df)} rows")
    return df

def _get_fetcher(sqlite_path: str, api_endpoint: str, timeout: int):
    use_sqlite = bool(sqlite_path and os.path.isfile(sqlite_path))
    if use_sqlite:
        return lambda sql: _fetch_sqlite(sql, sqlite_path)
    else:
        return lambda sql: _fetch_csv(sql, api_endpoint, timeout)

def _batched_fetch(template_sql: str,
                   id_list: list[int],
                   fetch,
                   chunk_size: int) -> pd.DataFrame:
    if not id_list:
        print("[BATCH] No IDs to fetch.")
        return pd.DataFrame()
    dfs = []
    for i in range(0, len(id_list), chunk_size):
        chunk = id_list[i : i + chunk_size]
        print(f"[BATCH] chunk {i//chunk_size+1}: {len(chunk)} IDs")
        sql = template_sql.format(ids=",".join(map(str, chunk)))
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


# ——— Part 1: library lookup ———

def get_library_table(
    smiles: str,
    searchtype: str = "exact",
    sqlite_path: str | None = None,
    api_endpoint: str = "http://127.0.0.1:8001/masst_records",
    timeout: int = 10
) -> pd.DataFrame:
    """
    Given a SMILES, returns the library_table for its InChIKey prefix.
    """
    if searchtype not in ["exact", "substructure"]:
        raise ValueError(f"Invalid search type: {searchtype}. Must be 'exact' or 'substructure'.")
    
    if searchtype == "exact":
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        prefix = inchi.MolToInchiKey(mol).split('-')[0]

        fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
        lib_sql = (
            "SELECT * FROM library_table "
            f"WHERE InChIKey_smiles_firstBlock = '{prefix}' "
            "AND ppmBetweenExpAndThMass <= 20 "
            "AND msMassAnalyzer NOT IN ('quadrupole', 'ion trap')"
        )
        library_df = fetch(lib_sql)

        if 'collision_energy' in library_df.columns:
            library_df['collision_energy'] = library_df['collision_energy'].apply(lambda x: str(x) if pd.notna(x) else 'unknown')

        # Only fillna for known string columns
        library_df[['Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']] = library_df[
            ['Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']
        ].fillna('unknown')

        
        # rename column InChIKey_smiles_firstBlock to inchikey_first_block
        if 'InChIKey_smiles_firstBlock' in library_df.columns:
            library_df.rename(columns={'InChIKey_smiles_firstBlock': 'inchikey_first_block'}, inplace=True)

        # rename spectrum_id to query_spectrum_id
        if 'spectrum_id' in library_df.columns:
            library_df.rename(columns={'spectrum_id': 'query_spectrum_id'}, inplace=True)

        df_final = library_df.copy()
    else:

        structure_type = detect_smiles_or_smarts(smiles)

        fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
        lib_sql_minimal = (
            "SELECT spectrum_id_int, Smiles, InChIKey_smiles FROM library_table "
            "WHERE ppmBetweenExpAndThMass <= 20 "
            "AND msMassAnalyzer NOT IN ('quadrupole', 'ion trap') "
            "AND Smiles IS NOT NULL "
            "AND Smiles != '' "
            "AND Smiles != 'NaN'"
        )
        library_df_minimal = fetch(lib_sql_minimal)

        library_df_minimal = fetch_and_match_smiles(library_df_minimal, smiles, match_type=searchtype, smiles_name='only',
                                             smiles_type=structure_type, formula_base='any', element_diff='any',
                                             max_by_grp=None, max_overall=None)
        

        # If no matches, return early
        if isinstance(library_df_minimal, list) or library_df_minimal.empty:
            print("No matching structures found in the library.")
            df_final = pd.DataFrame()
            return df_final
        
        matched_ids = library_df_minimal['spectrum_id_int'].dropna().astype(int).unique().tolist()

        lib_sql_template = (
            "SELECT spectrum_id_int, spectrum_id, Compound_Name, Ion_Mode, collision_energy, Adduct, "
            "msManufacturer, msMassAnalyzer, GNPS_library_membership "
            "FROM library_table WHERE spectrum_id_int IN ({ids})"
        )

        df_metadata = _batched_fetch(lib_sql_template, matched_ids, fetch, chunk_size=500)

        # join and return
        df_final = library_df_minimal.merge(df_metadata, on='spectrum_id_int', how='left')

        if 'collision_energy' in df_final.columns:
            df_final['collision_energy'] = df_final['collision_energy'].apply(lambda x: str(x) if pd.notna(x) else 'unknown')

        df_final[['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']] = df_final[
            ['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']
        ].fillna('unknown')



        df_final.rename(columns={'spectrum_id': 'query_spectrum_id'}, inplace=True)


    for col in ['spectrum_id_int']:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    return df_final


# ——— Part 2: MASST + ReDU lookup ———

def get_masst_and_redu_tables(
    library_df: pd.DataFrame,
    cosine_threshold: float = 0.7,
    matching_peaks: int = 5,
    sqlite_path: str | None = None,
    api_endpoint: str = "http://127.0.0.1:8001/masst_records",
    timeout: int = 10,
    chunk_size: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a non-empty library_df, returns (masst_df, redu_df).
    If library_df is empty, both returned DataFrames will be empty.
    """
    if library_df.empty:
        print("[PART 2] empty library_df → nothing to fetch")
        return pd.DataFrame(), pd.DataFrame()

    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)

    # — Masst table —
    sids = library_df['spectrum_id_int'].dropna().unique().tolist()
    print(f"[STEP 2] masst_table for {len(sids)} spectrum_id_ints")
    if not sids:
        return pd.DataFrame(), pd.DataFrame()
    masst_sql_tmpl = (
        "SELECT * FROM masst_table "
        "WHERE spectrum_id_int IN ({ids}) "
        f"AND cosine >= {cosine_threshold} "
        f"AND matching_peaks >= {matching_peaks}"
    )
    masst_df = _batched_fetch(masst_sql_tmpl, sids, fetch, chunk_size)
    if masst_df.empty:
        print("[STEP 2] no masst hits → exiting part 2")
        return pd.DataFrame(), pd.DataFrame()

    # — add MRI strings —
    mids = masst_df['mri_id_int'].dropna().unique().tolist()
    if mids:
        print(f"[STEP 3a] fetching mri strings for {len(mids)} mri_id_ints")
        mri_sql   = "SELECT mri_id_int, mri FROM mri_table WHERE mri_id_int IN ({ids})"
        mri_map   = _batched_fetch(mri_sql, mids, fetch, chunk_size)
        if not mri_map.empty:
            masst_df = masst_df.merge(mri_map, on='mri_id_int', how='left')
        else:
            masst_df['mri'] = None
    else:
        masst_df['mri'] = None

    # — add spectrum_id strings —
    print(f"[STEP 3b] merging spectrum_id for {len(sids)} spectrum_id_ints")
    spec_map = library_df[['spectrum_id_int', 'query_spectrum_id', 'Adduct', 'Compound_Name']].drop_duplicates()
    masst_df = masst_df.merge(spec_map, on='spectrum_id_int', how='left')

    # — ReDU table —
    if not mids:
        print("[STEP 4] no mri_ids → skipping redu")
        return masst_df, pd.DataFrame()

    print(f"[STEP 4] redu_table for {len(mids)} mri_id_ints")
    redu_sql_tmpl = "SELECT * FROM redu_table WHERE mri_id_int IN ({ids})"
    redu_df = _batched_fetch(redu_sql_tmpl, mids, fetch, chunk_size)

    return masst_df, redu_df