import numpy as np
from immutables import Map
import git
from pathlib import Path

import streamlit as st
from data.canada import Provinces

from batch import batch_run, BatchResult
from components.model import TechnologyAdoptionModel


def run_page():
    st.title("Batch Runs")
    st.write(
        """
        This page allows you to run multiple simulations at once.
        """
    )

    all_provinces = [p.value for p in list(Provinces)]

    with st.form("batch_run_form"):
        st.markdown("## Batch Run Parameters")
        col1, col2 = st.columns([1, 2])
        with col1:
            n_agents = st.multiselect("Select agents", [50, 1000, 200], 50)
            provinces = st.multiselect("Select provinces", all_provinces, Provinces.BC)
            interact = st.checkbox(
                "Do agents interact?",
            )
        with col2:
            start_year = st.slider("Start year", 2000, 2020, 2000)
            n_steps = st.slider(
                "Number of steps",
                50,
                200,
                80,
            )
            random_seed_range = st.select_slider(
                "Random seed range",
                options=np.linspace(0, 100, 100, dtype=int),
                value=(20, 28),
            )

        # st.columns(2)

        st.markdown("---")

        batch_parameters = {
            "N": n_agents,
            "province": provinces,  # , "Alberta", "Ontario"],
            "random_seed": range(*random_seed_range),
            "start_year": [start_year],
            "interact": [interact],
            "n_segregation_steps": [40],
        }

        run_model = st.form_submit_button("run models")

    if run_model:
        with st.expander("dev info"):
            st.markdown("# Running with the following parameters")
            col1, col2 = st.columns(2)
            with col1:
                for k, v in batch_parameters.items():
                    if isinstance(v, list):
                        batch_parameters[k] = tuple(v)
                batch_parameters["n_steps"] = n_steps
                st.write(batch_parameters)

            with col2:            
                repo = git.Repo(search_parent_directories=True)
                branch_name = str(repo.head.ref)
                batch_param_hash = hash(Map(batch_parameters))

                results_path = Path(f"results/{branch_name.replace('/','_')}").joinpath(str(batch_param_hash))
                st.markdown(
                    f"""
                            Results will be stored using the following pattern.
                            The current git branch is `{branch_name}`, 
                            and the hash of `batch_parameters` is `{batch_param_hash}`.
                            The results will be stored in `{results_path}`
                            """
                )

        if results_path.exists():
            # load the result
            st.write(f"loading {results_path}")
            b_result = BatchResult.from_directory(results_path)
        else:
            print(f"executing models for {results_path}")
            results_path.mkdir(parents=True)

            # remove n_steps, because that can't be passed to the model
            max_steps = batch_parameters.pop("n_steps")
            results = batch_run(
                TechnologyAdoptionModel,
                batch_parameters,
                number_processes=None,
                max_steps=max_steps,
                data_collection_period=1
            )
            b_result = BatchResult(results_path, batch_parameters, data=results)
            saved_files = b_result.save()
            st.write("saved:", saved_files)
        
        st.markdown("# Results")

        st.markdown("## Technology shares and attitudes over time")
        tech_shares_col, attitude_col = st.columns(2)
        with tech_shares_col:
            tech_shares = b_result.tech_shares_fig(show_legend=False)
            st.pyplot(tech_shares)

        with attitude_col:
            attitude_fig = b_result.attitudes_fig()
            attitude_fig.tight_layout()
            st.pyplot(attitude_fig)


        st.markdown("## Cumulative adoptions and their reason")
        st.markdown("""This figure shows the cumulative adoptions that have happened over time. 
                    Note, that individual cumulative adoptions don't ever have a \'dip\', but here the mean across several model runs is displayed.""")
        adoption_fig = b_result.adoption_details_fig()
        st.pyplot(adoption_fig)

        

if __name__ == "__main__":
    run_page()
