import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import importlib.util
from rdkit import Chem
from PIL import Image
import base64
import io
from bin.workflow_stepwise import retrieveSpectraCandidates, retrieve_raw_data_matches 
from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables
from bin.match_smiles import detect_smiles_or_smarts
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from urllib.parse import quote_plus
from formula_validation.Formula import Formula
import requests
import re

# datasette masst_records.sqlite --setting max_returned_rows 1000000 --setting sql_time_limit_ms 60000

# — load config —
config_path = "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


# Add a tracking token
html('<script async defer data-website-id="<your_website_id>" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)

# Write the page label
st.set_page_config(
    page_title="StructureMASST App", 
    layout="wide",
    page_icon="👋",
)

st.title("StructureMASST")

# — SMILES or CSV input —
col1, col_or, col2 = st.columns([4,2,4])
with col1:
    smiles_input = st.text_input("SMILES/SMARTS", placeholder="Enter SMILES or SMARTS")

with col_or:
    st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

with col2:
    uploaded_file = st.file_uploader("Drop CSV file for batch search", type=["csv"])

# — Display molecule if valid SMILES/SMARTS —
try:
    from rdkit.Chem import Draw
    def mol_to_base64_img(mol, size=(300, 300)):
        try:
            img = Draw.MolToImage(mol, size=size)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"<img src='data:image/png;base64,{img_str}' style='margin-top:1em;'/>"
        except Exception:
            return f"<p style='color:red;'>Failed to draw molecule image.</p>"
except ImportError:
    mol_to_base64_img = None


# --- render logic ---
if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input) or Chem.MolFromSmarts(smiles_input)
    if mol:
        if mol_to_base64_img:
            st.markdown(mol_to_base64_img(mol), unsafe_allow_html=True)
        else:
            st.info(f"SMILES: {smiles_input}")
    else:
        st.warning("Not a valid SMILES/SMARTS")


# — mode selection UI —
col_a1, col_b2, _ = st.columns([2,1,2])
with col_a1:
    searchtype_option = st.radio(
        "Find available MS/MS spectra", 
        ["exact structure match", "substructure match", "tanimoto similarity"], 
        horizontal=True
    )

if searchtype_option == "tanimoto similarity":
    with col_b2:
        st.text_input("Tanimoto threshold", value="0.8", key="tanimoto_threshold")

# Map UI option to backend value
if searchtype_option == "exact structure match":
    searchtype_option = "exact"
elif searchtype_option == "substructure match":
    searchtype_option = "substructure"
elif searchtype_option == "tanimoto similarity":
    searchtype_option = "tanimoto"


# — run the search —
if st.button("Check Available Spectra"):

    # Reset upstream & downstream state
    for key in [
        "selected_queries",
        "df_library_conflicts",
        "grouped_results",
        "raw_results",
        "molecule_overview"
    ]:
        st.session_state.pop(key, None)

    # initialize what's needed 
    st.session_state.selected_queries = {}
    st.session_state.df_library_conflicts = None
    st.session_state.grouped_results = {}
    st.session_state.molecule_overview = {}

    # organize input structure queries
    smiles_list = []
    if smiles_input:
        smiles_list = [smiles_input]
        name_list = ['Input Query']
    elif uploaded_file is not None:
        df_in = pd.read_csv(uploaded_file)
        if "smiles" in df_in.columns and "name" in df_in.columns:
            smiles_list = df_in["smiles"].dropna().tolist()
            name_list = df_in["name"].dropna().tolist()
        else:
            st.warning("CSV must contain a 'smiles' and 'name' column.")
            st.stop()
    else:
        st.warning("Please enter a SMILES or upload a CSV.")
        st.stop()

    # process each input structure query separately to retrieve spectra
    grouped_results = defaultdict(dict)
    molecule_overview = defaultdict(dict)
    for smi, name in zip(smiles_list, name_list):
        try:
            df_library_structurematch = get_library_table(
                smiles=smi,
                searchtype=searchtype_option,
                sqlite_path=config.PATH_TO_SQLITE,
                api_endpoint=config.MASSTRECORDS_ENDPOINT,
                timeout=config.MASSTRECORDS_TIMEOUT
            )
            print(f"Retrieved {len(df_library_structurematch)} records from MASSTrecords")
            if df_library_structurematch.empty:
                st.info(f"No library entries found for: {smi}")
                continue
            # Setup for library conflicts. this is currently not doing anything
            df_library_conflicts = pd.DataFrame()
            df_library_conflicts["inchikey_first_block"] = pd.Series(dtype=str)
        except Exception as e:
            st.error(f"Error for {smi}: {e}")

          

        # Setup for library conflicts. this is currently not doing anything
        print(f"Processing {name} with {len(df_library_structurematch)} matches")
        overview = []
        for ik in df_library_structurematch["inchikey_first_block"].unique():
            sub_struct = df_library_structurematch[df_library_structurematch["inchikey_first_block"] == ik].copy()
            sub_conf   = df_library_conflicts[df_library_conflicts["inchikey_first_block"] == ik].copy()
            grouped_results[name][ik] = {"structure": sub_struct, "conflicts": sub_conf}
            st.session_state.selected_queries[ik] = list(sub_struct["query_spectrum_id"].unique())


            # pick most common Compound_Name (tie-break by len≈20 & fewest special chars)
            names = sub_struct["Compound_Name"].dropna().astype(str)
            if not names.empty:
                vc = names.value_counts()
                top = vc.iloc[0]
                cands = vc[vc == top].index.tolist()
                def special_count(s): 
                    return len(re.findall(r"[^A-Za-z0-9]", s))
                best_name = min(cands, key=lambda s: (abs(len(s) - 20), special_count(s)))
            else:
                best_name = ""

            # grab first SMILES
            smiles = sub_struct["Smiles"].dropna().astype(str)
            inchikey_first_block = sub_struct["inchikey_first_block"].dropna().astype(str)
            first_smi = smiles.iloc[0] if not smiles.empty else ""
            ikb = inchikey_first_block.iloc[0] if not inchikey_first_block.empty else ""

            overview.append({
                "Compound_Name": best_name,
                "inchikey_first_block": ikb,
                "Smiles": first_smi
            })

        st.session_state.molecule_overview[name] = pd.DataFrame(overview)




    # get results into session state
    st.session_state.grouped_results = grouped_results


# render outer tabs for each structure query
if "grouped_results" in st.session_state and st.session_state["grouped_results"]:

    name_tabs = st.tabs(list(st.session_state.grouped_results.keys()))
    
    with st.expander("Available Library Entries", expanded=True):
        st.markdown("### Available Library Entries")
        name_tabs = st.tabs(list(st.session_state.grouped_results.keys()))
        for name, name_tab in zip(st.session_state.grouped_results.keys(), name_tabs):
            with name_tab:

                tables = st.session_state.grouped_results[name]

                # number of molecules = number of 2d inchikey keys
                num_molecules = len(tables)

                # total number of matches across all "structure" tables
                total_matches = sum(
                    tbl["structure"].shape[0]
                    for tbl in tables.values()
                )

                # messages on retrieved spectra
                st.markdown(
                    f"##### Retrieved **{total_matches}** spectra for **{num_molecules}** molecule(s) ({name}).",
                )

                st.markdown(
                    "<div style='font-size:0.9em;'>"
                    "Below you see the public MS/MS spectra available for your search. The higher the spectral diversity, the less blind-spots you will have in public metabolomics raw data."
                    "</div>",
                    unsafe_allow_html=True
                )

                if total_matches == 0:
                    st.info("No spectra available for this structure.")
                    continue
                # Sankey per query structure
                ##########

                # concatenate all the 'structure' dfs under this name
                df_all = pd.concat(
                    [v["structure"] for v in st.session_state.grouped_results[name].values()],
                    ignore_index=True
                )

                # Check if all required columns exist before proceeding
                required_cols = ["msMassAnalyzer", "Ion_Mode", "Adduct", "collision_energy"]
                missing_cols = [col for col in required_cols if col not in df_all.columns]
                if not missing_cols:

                    # Fixing levels
                    stages = ["msMassAnalyzer", "Ion_Mode", "Adduct", "collision_energy"]
                    df_sankey = df_all[stages].dropna()

                    # Create labels for Sankey diagram
                    labels = []
                    for col in stages:
                        labels += df_sankey[col].unique().tolist()
                    labels = list(dict.fromkeys(labels))

                    # get up to 5 colors (should not be more different instrument types than that)
                    ms_cats = df_sankey["msMassAnalyzer"].unique().tolist()
                    palette = px.colors.qualitative.Safe[:5]  
                    color_map = {cat: palette[i] for i, cat in enumerate(ms_cats)}

                    # build links
                    source, target, value, link_colors = [], [], [], []
                    for i in range(len(stages) - 1):
                        for cat in ms_cats:
                            df_cat = df_sankey[df_sankey["msMassAnalyzer"] == cat]
                            grp = (
                                df_cat
                                .groupby([stages[i], stages[i + 1]])
                                .size()
                                .reset_index(name="count")
                            )
                            for _, row in grp.iterrows():
                                source.append(labels.index(row[stages[i]]))
                                target.append(labels.index(row[stages[i + 1]]))
                                value.append(row["count"])
                                link_colors.append(color_map[cat].replace("rgb", "rgba").replace(")", f", {0.3})"))


                    # all nodes light grey with black border
                    node_colors = ["#F2F2F2"] * len(labels)

                    fig = go.Figure(
                        go.Sankey(
                            arrangement="snap",
                            # ← trace‑level label styling
                            textfont=dict(family="Arial, sans-serif", size=12, color="black"),

                            node=dict(
                                label=labels,
                                color=node_colors,
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                            ),
                            link=dict(
                                source=source,
                                target=target,
                                value=value,
                                color=link_colors,
                            ),
                        )
                    )

                    # add stage annotations
                    fig.update_layout(
                        font=dict(family="Arial, sans-serif", size=12),
                        margin=dict(l=60, r=60, t=120, b=20),
                    )

                    # add stage labels above the Sankey diagram
                    stage_labels = ["Mass Analyzer", "Ion Mode", "Adduct", "Collision Energy"]

                    # calculate x positions for each stage label
                    n = len(stage_labels) - 1
                    for i, label in enumerate(stage_labels):
                        x = i / n
                        # xanchor depends on position
                        if i == 0:
                            xanchor = "left"
                        elif i == n:
                            xanchor = "right"
                        else:
                            xanchor = "center"

                        # add annotation for each stage label
                        fig.add_annotation(
                            x=x,
                            y=1.02,
                            xref="paper",
                            yref="paper",
                            text=label,
                            showarrow=False,
                            font=dict(size=14, color="black"),
                            xanchor=xanchor
                        )
                    # update layout
                    st.plotly_chart(fig, use_container_width=True)



                molecule_overview_df = st.session_state.molecule_overview[name]
                
                # display it with clickable links 
                table_mol = st.dataframe(
                    molecule_overview_df,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="multi-row",
                    use_container_width=True,
                    key=f"{name}_molecule_table"
                )


                # grab list of selected row indices
                selected = table_mol.selection.rows

                # show buttons with actions for selected molecules
                col1_mol_level, col2_mol_level, _ = st.columns([2, 2, 6])
                with col1_mol_level:
                    if st.button("Remove selected molecule(s)", key=f"{name}_mol_remove"):
                        if selected:
                            molecule_overview_df = molecule_overview_df.drop(molecule_overview_df.index[selected])
                            st.session_state.molecule_overview[name] = molecule_overview_df
                            unique_inchikeys = molecule_overview_df["inchikey_first_block"].unique().tolist()
                            # also remove from grouped_results from same name
                            st.session_state.grouped_results[name] = {
                                ik: data
                                for ik, data in st.session_state.grouped_results[name].items()
                                if ik in unique_inchikeys
                            }

                        else:
                            st.warning("No rows selected!")
                        st.rerun()

                with col2_mol_level:
                    if st.button("Keep only selected molecule(s)", key=f"{name}_mol_keep"):
                        if selected:
                            molecule_overview_df = molecule_overview_df.iloc[selected].reset_index(drop=True)
                            st.session_state.molecule_overview[name] = molecule_overview_df
                            unique_inchikeys = molecule_overview_df["inchikey_first_block"].unique().tolist()
                            # also remove from grouped_results from same name
                            st.session_state.grouped_results[name] = {
                                ik: data
                                for ik, data in st.session_state.grouped_results[name].items()
                                if ik in unique_inchikeys
                            }
                        else:
                            st.warning("No rows selected!")
                        st.rerun()

                # update the session state with the filtered dataframe
                st.session_state.molecule_overview[name] = molecule_overview_df
                



                # display each query structures available spectra as table
                with st.expander("Molecules by InChIKey", expanded=True):

                    # create a tab for each 2d InChIKey
                    ik_tabs = st.tabs(list(st.session_state.grouped_results[name].keys()))
                    for ik, ik_tab in zip(st.session_state.grouped_results[name].keys(), ik_tabs):
                        with ik_tab:

                            data = st.session_state.grouped_results[name][ik]
                            df0 = data["structure"].copy()

                            # create spectrum link
                            df0["spectrum_link"] = (
                                "http://metabolomics-usi.gnps2.org/dashinterface?usi1=mzspec%3AGNPS%3AGNPS-LIBRARY%3Aaccession%3A"
                                + df0["query_spectrum_id"].astype(str)
                                + "&width=10.0&height=6.0&mz_min=None&mz_max=None&max_intensity=125&annotate_precision=4&annotation_rotation=90&cosine=standard&fragment_mz_tolerance=0.02&grid=True&annotate_peaks=%5B%5B%5D%2C%20%5B%5D%5D"
                            )

                            # Make the spectrum column the first column
                            df0 = df0[["spectrum_link"] + [col for col in df0.columns if col != "spectrum_link"]]

                            # display it with clickable links 
                            table_evt = st.dataframe(
                                df0,
                                column_config={
                                    "spectrum_link": st.column_config.LinkColumn(
                                        label="USI Viewer",
                                        display_text="Open Spectrum"
                                    )
                                },
                                hide_index=True,
                                on_select="rerun",
                                selection_mode="multi-row",
                                use_container_width=True,
                                key=f"{name}_{ik}_table"
                            )

                            # grab list of selected row indices
                            selected = table_evt.selection.rows

                            # show buttons with actions for selected spectra
                            col1, col2, _ = st.columns([2, 2, 6])
                            with col1:
                                if st.button("Remove selected spectra", key=f"{name}_{ik}_remove"):
                                    if selected:
                                        df_filtered = df0.drop(df0.index[selected])
                                        st.session_state.grouped_results[name][ik]["structure"] = df_filtered
                                    else:
                                        st.warning("No rows selected!")
                                    st.rerun()

                            with col2:
                                if st.button("Keep only selected spectra", key=f"{name}_{ik}_keep"):
                                    if selected:
                                        df_filtered = df0.iloc[selected].reset_index(drop=True)
                                        st.session_state.grouped_results[name][ik]["structure"] = df_filtered
                                    else:
                                        st.warning("No rows selected!")
                                    st.rerun()

                            # update the session state with the filtered dataframe
                            st.session_state.grouped_results[name][ik]["structure"] = df0

        
    
    # selection menu for raw data search
    col_a, col_b = st.columns(2)
    with col_a:
        option = st.radio(
            "Mode",
            ["FASSTrecords", "FASST"],
            horizontal=True,
            key="mode"
        )
    with col_b:
        st.empty()

    last_iteration = "<DATE>"  # need to get version into sql table

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="
            border-left: 4px solid #2c7be5;
            padding: 1em;
            margin: 0.5em 0;
            background-color: #f0f8ff;
            border-radius: 4px;
        ">
        <h4 style="margin:0 0 0.5em;">
            <strong>FASSTrecords</strong>
        </h4>
        <p style="margin:0; line-height:1.5; font-size:0.95em;">
            This is <strong>very fast</strong>: obtain distributions of thousands of molecules in seconds,<br/>
            with substructure‑enabled search.<br/>
            Based on pre‑computed spectral matching scores of annotated spectra.<br/>
            Last iteration: <strong>{last_iteration}</strong>.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            border-left: 4px solid #e76f51;
            padding: 1em;
            margin: 0.5em 0;
            background-color: #fff5f0;
            border-radius: 4px;
        ">
        <h4 style="margin:0 0 0.5em;">
            <strong>FASST</strong>
        </h4>
        <p style="margin:0; line-height:1.5; font-size:0.95em;">
            Can be rather slow—a maximum of <strong>10–50 spectra</strong> per query is recommended.<br/>
            Each search may take a few minutes, depending on traffic.<br/>
            Allows modification searches.<br/>
            Always up to date with the latest raw data indexed at
            <a href="https://fasst.gnps2.org/" target="_blank" style="color:#e76f51;">fasst.gnps2.org</a>.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cosine and Matching Peaks input 
    col3, col4, _, _ = st.columns(4)
    with col3:
        min_cosine = st.text_input("Minimum Cosine", value="0.7")
    with col4:
        min_peaks = st.text_input("Minimum Matching Peaks", value="5")

    # Conditional input for FASST 
    if option == "FASST":
        col5, col6, _, _ = st.columns(4)
        with col6:
            prec_tol = st.text_input("Precursor Tolerance (Da)", value="0.02")
            do_modification_search = st.checkbox("Modification search", value=False)


        with col5:
            frag_tol = st.text_input("Fragment Tolerance (Da)", value="0.02")
            if do_modification_search:
                col_elim, col_add = st.columns(2)
                with col_elim:
                    do_elimination = st.checkbox("Elimination search", value=True)
                with col_add:
                    do_addition = st.checkbox("Addition search", value=True)
                sub_col1, col_or, sub_col2 = st.columns([2,0.5,2])
                with sub_col1:
                    modification_formula = st.text_input("Modification formula", placeholder="O for hydroxylation")
                with col_or:
                    st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)
                with sub_col2:
                    modification_mass = st.text_input("Modification mass (Da)", placeholder="15.9949 for O")
                do_subsetModificationSearch = st.checkbox("Only report modified molecules if <condition>", value=False)

                if do_subsetModificationSearch:
                    list_of_values = ["Raw file", "ATTRIBUTE_DatasetAccession", "NCBITaxonomy"]
                    modification_condition = st.selectbox("Unmodified found in same", options=list_of_values)

    ctrl1, ctrl2, _ = st.columns([1,1,7])
    with ctrl1:
        do_search = st.button("Search Raw Data")


    # perform the raw data search
    if do_search:
        new_results = {}

        if option == "FASSTrecords":
            # build each queries aggregated table
            for name, ik_dict in st.session_state.grouped_results.items():
                sel_frames = []
                for ik, data in ik_dict.items():
                    
                    df_struct = data["structure"]
                    if not df_struct.empty:
                        sel_frames.append(df_struct)

                if not sel_frames:
                    st.warning(f"No entries left for **{name}**, skipping.")
                    continue

                df_for_name = pd.concat(sel_frames, ignore_index=True)

                # get raw data from masstrecords
                masst_df, redu_df = get_masst_and_redu_tables(df_for_name,
                                                            cosine_threshold=float(min_cosine),
                                                            matching_peaks=int(min_peaks),
                                                            sqlite_path=config.PATH_TO_SQLITE,
                                                            api_endpoint=config.MASSTRECORDS_ENDPOINT,
                                                            timeout=config.MASSTRECORDS_TIMEOUT)

                # if cosine not in masst_df.columns return empty dataframes
                if "cosine" not in masst_df.columns or "matching_peaks" not in masst_df.columns:
                    new_results[name] = {"masst": pd.DataFrame(), "redu": pd.DataFrame()}
                    continue

                # subset results for sample matches table to best match by sample
                df_masst_sorted = masst_df.sort_values(by=["cosine", "matching_peaks"], ascending=[False, False])
                df_masst_unique = df_masst_sorted.drop_duplicates(subset="mri_id_int", keep="first")

                # add query spectrum ID and scan ID to redu_df //could potentially move this into get_masst_and_redu_tables
                redu_df = redu_df.merge(
                    df_masst_unique[["mri_id_int", "scan_id", "query_spectrum_id", 'Adduct', 'Compound_Name', 'inchikey_first_block']],
                    on="mri_id_int",
                    how="left"
                )

                new_results[name] = {"masst": masst_df, "redu": redu_df}
            
        elif option == "FASST":

            # build each queries aggregated table
            for name, ik_dict in st.session_state.grouped_results.items():
                sel_frames = []
                for ik, data in ik_dict.items():
                    df_struct = data["structure"]
                    if not df_struct.empty:
                        sel_frames.append(df_struct)

                if not sel_frames:
                    st.warning(f"No entries left for **{name}**, skipping.")
                    continue

                df_for_name = pd.concat(sel_frames, ignore_index=True)

                # if we do modification and got a molecular formula calculate the monoisotopic mass of the expected mass difference
                formulaModi_object = Formula.formula_from_str(modification_formula) if do_modification_search and modification_formula else None
                try:
                    modification_mass = formulaModi_object.get_monoisotopic_mass()
                except AttributeError:
                    modification_mass = modification_mass if 'modification_mass' in locals() else None

                # retrieve raw data matches through MASST
                masst_df, redu_df = retrieve_raw_data_matches(
                    df_for_name,
                    database='metabolomicspanrepo_index_nightly',
                    precursor_mz_tol=float(prec_tol),
                    fragment_mz_tol=float(frag_tol),
                    min_cos=float(min_cosine),
                    matching_peaks=int(min_peaks),
                    analog=do_modification_search,
                    modimass=modification_mass,
                    elimination=do_elimination if 'do_elimination' in locals() else False,
                    addition=do_addition if 'do_addition' in locals() else False,
                    modification_condition=modification_condition if 'modification_condition' in locals() else None,
                )
                
                new_results[name] = {"masst": masst_df, "redu": redu_df}

        # store the results in session state
        st.session_state.raw_results = new_results

    # display results in tabs
    if st.session_state.get("raw_results"):


        raw_results = st.session_state.get("raw_results", {})

        # print lens of what has been found
        for name, df_pair in raw_results.items():
            # count unique MRI IDs in the 'redu' dataframe
            num_unique_mri = df_pair['redu']['mri_id_int'].dropna().nunique()

            st.markdown(
                f"##### Found **{len(df_pair['masst'])}** spectral hits and **{num_unique_mri}** unique matching samples with ReDU metadata for **{name}**."
            )

        has_valid_results = any(
            not df_pair["masst"].empty or not df_pair["redu"].empty
            for df_pair in raw_results.values()
        )

        if has_valid_results:
            with st.expander("Metabolomics Raw Data Matches", expanded=True):
                st.markdown("### Metabolomics Raw Data Matches")

                # create tabs for each query structure
                result_tabs = st.tabs(list(st.session_state.raw_results.keys()))
                for name, tab in zip(st.session_state.raw_results.keys(), result_tabs):
                    with tab:

                        # get the results for this query structure
                        st.markdown(f"##### Retrieved **{len(st.session_state.raw_results[name]['masst'])}** spectral hits for {name} with **{len(st.session_state.raw_results[name]['redu'])}** matching samples with ReDU metadata.")

                        st.markdown(
                            "<div style='font-size:0.9em;'>"
                            "Below you can see the the raw data matches available to you. Keep in mind that MS/MS annotations, while often accurate, are not perfect. This is the starting pointing of your analysis, not the end."
                            "</div>",
                            unsafe_allow_html=True
                        )

                        # create two subtabs for this query structure
                        sub_tabs = st.tabs(["Sample matches", "Spectral matches"])

                        
                        # redu matches tab
                        with sub_tabs[0]:
                            df_redu = st.session_state.raw_results[name]["redu"]
                            column_options = df_redu.columns.tolist()


                            # Make sankey diagram
                            ##########

                            # define defaults
                            default_vals = [
                                "ATTRIBUTE_DatasetAccession",
                                "UBERONBodyPartName",
                                "NCBIDivision",
                                "NCBITaxonomy",
                            ]
                            def_val = lambda v: v if v in column_options else column_options[0]

                            # initialize session_state defaults 
                            for i, default in enumerate(default_vals, start=1):
                                key = f"{name}_col{i}"
                                if key not in st.session_state:
                                    st.session_state[key] = def_val(default)

                            # make four selectboxes 
                            col1_c, col2_c, col3_c, col4_c = st.columns(4)
                            with col1_c:
                                st.selectbox(
                                    "Column 1",
                                    column_options,
                                    index=column_options.index(st.session_state[f"{name}_col1"]),
                                    key=f"{name}_col1"
                                )
                            with col2_c:
                                st.selectbox(
                                    "Column 2",
                                    column_options,
                                    index=column_options.index(st.session_state[f"{name}_col2"]),
                                    key=f"{name}_col2"
                                )
                            with col3_c:
                                st.selectbox(
                                    "Column 3",
                                    column_options,
                                    index=column_options.index(st.session_state[f"{name}_col3"]),
                                    key=f"{name}_col3"
                                )
                            with col4_c:
                                st.selectbox(
                                    "Column 4",
                                    column_options,
                                    index=column_options.index(st.session_state[f"{name}_col4"]),
                                    key=f"{name}_col4"
                                )

                            # pull the latest values and build your Sankey immediately
                            col1 = st.session_state[f"{name}_col1"]
                            col2 = st.session_state[f"{name}_col2"]
                            col3 = st.session_state[f"{name}_col3"]
                            col4 = st.session_state[f"{name}_col4"]

                            # build Sankey
                            df = df_redu.copy()
                            for col in (col1, col2, col3, col4):
                                top10 = df[col].value_counts().nlargest(10).index
                                df[col + "_s"] = df[col].where(df[col].isin(top10), "others")

                            stages = [col1 + "_s", col2 + "_s", col3 + "_s", col4 + "_s"]
                            df["color_key"] = df[stages[0]]

                            labels = []
                            for i, stg in enumerate(stages, start=1):
                                uniques = df[stg].dropna().unique().tolist()
                                labels += [f"{i}_{u}" for u in uniques]
                            labels = list(dict.fromkeys(labels))

                            idx = {}
                            for i, stg in enumerate(stages, start=1):
                                for u in df[stg].dropna().unique():
                                    idx[(stg, u)] = labels.index(f"{i}_{u}")

                            column_1_vals = df[stages[0]].unique().tolist()
                            palette = px.colors.qualitative.Safe[:11]
                            color_map = {cat: palette[i] for i, cat in enumerate(column_1_vals)}

                            source, target, value, link_colors = [], [], [], []
                            for i in range(len(stages) - 1):
                                grp = (
                                    df
                                    .dropna(subset=[stages[i], stages[i+1], "color_key"])
                                    .groupby([stages[i], stages[i+1], "color_key"])
                                    .size()
                                    .reset_index(name="count")
                                )
                                for _, row in grp.iterrows():
                                    src_val = row[stages[i]]
                                    tgt_val = row[stages[i+1]]
                                    color_key = row["color_key"]
                                    source.append(idx[(stages[i], src_val)])
                                    target.append(idx[(stages[i+1], tgt_val)])
                                    value.append(row["count"])
                                    link_colors.append(color_map.get(color_key, "rgba(0,0,0,0.3)"))

                            fig = go.Figure(go.Sankey(
                                textfont=dict(family="Arial, sans-serif", size=12, color="black"),
                                arrangement="snap",
                                node=dict(
                                    label=labels,
                                    color=["#F2F2F2"] * len(labels),
                                    pad=15,
                                    thickness=20,
                                    line=dict(color="black", width=0.5),
                                ),
                                link=dict(
                                    source=source,
                                    target=target,
                                    value=value,
                                    color=link_colors
                                ),
                            ))

                            # Add stage annotations
                            stage_names = [col1, col2, col3, col4]
                            n = len(stage_names) - 1
                            for i, name_ in enumerate(stage_names):
                                x = i / n
                                xanchor = "left" if i == 0 else "right" if i == n else "center"
                                fig.add_annotation(
                                    x=x, y=1.02, xref="paper", yref="paper",
                                    text=name_, showarrow=False,
                                    font=dict(size=14, color="black"),
                                    xanchor=xanchor
                                )

                            fig.update_layout(
                                font=dict(family="Arial, sans-serif", size=12),
                                margin=dict(l=60, r=60, t=120, b=20),
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # fig.write_image("./output/rawData_sankey.pdf", format="pdf", width=1240, height=400, scale=2)


                            raw_data_sankey_triggered = True


                            # sample matches tab
                            #########

                            # make library usis for the links
                            df_redu["lib_usi"] = df_redu["query_spectrum_id"].apply(
                                lambda x: (
                                    f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}" if x.startswith("CCMSLIB")
                                    else f"mzspec:MASSBANK::accession:{x}" if x.startswith("MSBNK")
                                    else x
                                )
                            )

                            # in every row add USI + :scan: + scan_id (as str)
                            df_redu["lib_usi"] = df_redu["lib_usi"] + ":scan:" + df_redu["scan_id"].astype(str)

                            # build links for best spectral match and modification site
                            def build_spectraresolver_link(row):
                                usi1 = quote_plus(f"{row['USI']}")
                                usi2 = quote_plus(row['lib_usi'])
                                return (
                                    f"http://metabolomics-usi.gnps2.org/dashinterface"
                                    f"?usi1={usi1}"
                                    f"&usi2={usi2}"
                                    f"&width=10.0&height=6.0&mz_min=None&mz_max=None"
                                    f"&max_intensity=125&annotate_precision=4&annotation_rotation=90"
                                    f"&cosine=standard&fragment_mz_tolerance=0.05"
                                    f"&grid=True&annotate_peaks=%5B%5B%5D%2C%20%5B%5D%5D"
                                )

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
                            
                           
                            df_redu["best_spectral_match"] = df_redu.apply(build_spectraresolver_link, axis=1)

                            if 'Modified' in df_redu.columns:
                                df_redu["modification_site"] = df_redu.apply(
                                    lambda row: build_modifinder_link(row)
                                    if (row["Modified"] != "no" and row["Adduct"] in 
                                        ['[M+H]1+', '[M-H]1', '[M+Na]1+', '[M+NH4]1+', '[M+K]1+', '[M+Cl]1-', '[M+Br]1-'])
                                    else '',
                                    axis=1
                                )


                            # drop columns used for link generation
                            df_redu = df_redu.drop(columns=["lib_usi"], errors="ignore")
                            df_redu = df_redu.drop(columns=["query_smiles"], errors="ignore")

                            # Reorder columns: best_spectral_match first, then modification_site if it exists, then the rest
                            cols = ["best_spectral_match"]
                            if "modification_site" in df_redu.columns:
                                cols.append("modification_site")
                            cols += [col for col in df_redu.columns if col not in cols]
                            df_redu = df_redu[cols]

                            # If modification column exists sort so that modified matches come first
                            if 'Modified' in df_redu.columns:
                                df_redu['Modified'] = df_redu['Modified'].astype(str).str.lower()
                                df_redu.loc[~df_redu['Modified'].isin(['addition', 'elimination', 'no']), 'Modified'] = pd.NA
                                df_redu['Modified'] = pd.Categorical(
                                    df_redu['Modified'],
                                    categories=['addition', 'elimination', 'no'],
                                    ordered=True
                                )
                                df_redu = df_redu.sort_values(by='Modified', ascending=True)


                            column_config = {
                                        "best_spectral_match": st.column_config.LinkColumn(
                                            label="best_spectral_match",
                                            display_text="View MS/MS match"
                                        )
                                    }
                            
                            if 'modification_site' in df_redu.columns:
                                column_config["modification_site"] = st.column_config.LinkColumn(
                                    label="Modification Site",
                                    display_text="View Modification Site"
                                )

                            # show dataframe
                            table_evt = st.dataframe(
                                df_redu,
                                column_config=column_config,
                                hide_index=True,
                                use_container_width=True,
                                on_select="rerun",
                                selection_mode="multi-row",
                                key=f"{name}_redu_table",
                            )

                            # grab the selected row positions
                            selected = table_evt.selection.rows

                            # buttons for selected rows
                            btn_col1, btn_col2, _ = st.columns([2,2,6])
                            with btn_col1:
                                if st.button("Remove selected rows", key=f"{name}_redu_remove"):
                                    if selected:
                                        st.session_state.raw_results[name]["redu"] = (
                                            df_redu.drop(df_redu.index[selected])
                                        )
                                    else:
                                        st.warning("No rows selected!")
                                    st.rerun()
                            with btn_col2:
                                if st.button("Keep only selected rows", key=f"{name}_redu_keep"):
                                    if selected:
                                        st.session_state.raw_results[name]["redu"] = (
                                            df_redu.iloc[selected].reset_index(drop=True)
                                        )
                                    else:
                                        st.warning("No rows selected!")
                                    st.rerun()
                                                
                        # spectral matches tab
                        with sub_tabs[1]:
                            st.subheader(f"Spectral matches for {name}")
                            df_masst = st.session_state.raw_results[name]["masst"]
                            st.dataframe(df_masst, use_container_width=True)
        
        else:
            st.warning("No raw data matches found. Please try a different query structure or adjust your search parameters.")

    else:
        st.markdown("")
            

