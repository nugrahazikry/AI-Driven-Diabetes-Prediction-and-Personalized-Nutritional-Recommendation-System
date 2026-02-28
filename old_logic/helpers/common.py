"""
Common Helper Functions
=======================
Utility functions for page navigation and state management.
"""

import streamlit as st


def switch_page(page_name):
    """
    Switch to a different page in the Streamlit app.
    
    Args:
        page_name: Name of the page to switch to
    """
    st.session_state.page = page_name
    st.rerun()


def update_kelamin(selected_kelamin):
    """
    Update gender selection in session state.
    
    Args:
        selected_kelamin: Selected gender value
    """
    if st.session_state.jenis_kelamin != selected_kelamin:
        st.session_state.jenis_kelamin = selected_kelamin
