import numpy as np
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
                st.write(batch_parameters)

            with col2:            

                results_path = BatchResult.get_results_dir(batch_parameters)
                st.markdown(
                    f"""
                            Results will be stored using the following pattern: `results/<GIT_BRANCH>/<PARAMETER_HASH>`.
                            The results will be stored in `{results_path}`
                            """
                )
        
        b_result = BatchResult.from_parameters(batch_parameters, max_steps=n_steps)
        saved_to = b_result.save()
        st.write(f"Saved results to {saved_to}")
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

        incr_adopt_fig = b_result.adoption_details_fig_facet(n_facet_cols=3)
        st.plotly_chart(incr_adopt_fig)

        

if __name__ == "__main__":
    run_page()
