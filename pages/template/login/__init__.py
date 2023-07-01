import streamlit as st

from utils.db import validate_login
from utils.session import get_state
from pages.template.se_landing import se_landing
from pages.template.admin import admin_landing

def logout():
    st.session_state["login"] = {}
    st.experimental_rerun()

def login_screen():
    if get_state(st, "login"):
        st.sidebar.button("Logout", on_click=logout)
        st.sidebar.header("Speech Editor")

    if not st.session_state.get("login"):
        print("here", st.session_state.get("login"))
        with st.form(key="login_form"):
            st.subheader("Login")
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                login_details = validate_login(username, password)
                if login_details:
                    if login_details["deactivated"]:
                        st.warning("User Deactivated, please contact Admin")
                    else:    
                        st.session_state["login"] = {}
                        st.session_state["login"]["user_logged_in"] = True
                        st.session_state["login"]["username"] = login_details["username"]
                        st.session_state["login"]["name"] = login_details["name"]
                        st.session_state["login"]["user_type"] = login_details["user_type"]
                        st.experimental_rerun()
                else:
                    st.warning("Incorrect Username or Password")
    else:
        if st.session_state["login"]["user_type"] == "ADMIN":
            admin_landing()
        elif st.session_state["login"]["user_type"] == "ANNOTATOR":
            se_landing()

