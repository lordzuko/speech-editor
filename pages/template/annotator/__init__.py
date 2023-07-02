from os import write

import streamlit as st
from pages.template.se_landing import se_ui
from streamlit.proto.Selectbox_pb2 import Selectbox
from utils.db import update_password
from utils.models import Projects

from .queries import fetch_available_projects

annotation_project = {
    "se": se_ui,
}

def annotator_landing():
    task = st.sidebar.selectbox(
        "Manage Account", ["", "Change Password", "Annotation Management"]
    )
    if task == "Change Password":
        st.subheader("Change Password")

        with st.form(key="change_password", clear_on_submit=True):
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input(
                "Confirm New Password", type="password"
            )

            submit_button = st.form_submit_button(label="Change Password")

        if submit_button:
            if new_password == confirm_new_password:
                if update_password(
                    st.session_state["login"]["username"], new_password
                ):
                    st.success("Password Updated Successfully")
                else:
                    st.warning("Account is not registered yet!")
            else:
                st.warning("Confirm password don't match, please check")

    elif task == "Annotation Management":
        if st.session_state["login"]["user_type"] == "ADMIN":
            annotation_task = st.sidebar.selectbox(
                "Annotation",
                [
                    "",
                    "Assigned Annotation Tasks",
                    "Check Tagging Stats",
                ],
            )
        else:
            annotation_task = st.sidebar.selectbox(
                "Annotation", ["", "Assigned Annotation Tasks", "Check Tagging Stats"]
            )
        if annotation_task == "Assigned Annotation Tasks":
            # st.markdown("### Select annotation project from left sidebar")
            projects = fetch_available_projects(st.session_state["login"]["username"])
            project_mapping = {x["project_name"]: x["project_id"] for x in projects}
            project_names = [x["project_name"] for x in projects]
            selected_project = st.sidebar.selectbox(
                "Select Annotation Project", [""] + project_names
            )
            pid = project_mapping.get(selected_project, None)
            if pid == "se":
                annotation_project.get(pid)()
        elif annotation_task == "Check Tagging Stats":
            st.markdown("### Feature Coming Soon!")

        else:
            st.markdown("### Please select annotation project from the left sidebar")
    else:
        st.header("Annotator's Dashboard")
        st.markdown("---")
        st.write("Please check select the tasks to be performed from the left sidebar")
