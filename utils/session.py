def init_session_state(st, key, value):
    """
    Setup the session state
    """
    if key not in st.session_state:
        st.session_state[key] = value

def get_state(st, key):
    """
    Get Values from session state
    """
    # if key not in st.session_state:
    #     st.error("Internal error: `{}` not present in session_state.".format(key))

    return st.session_state.get(key)
