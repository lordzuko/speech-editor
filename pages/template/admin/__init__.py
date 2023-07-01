
import datetime

import pandas as pd
import streamlit as st
from utils import make_hashes
from utils.db import delete_account, update_password, fetch_annotators
from utils.models import Users, Projects
from streamlit_tags import st_tags
from pages.template.annotator import annotator_landing

def admin_landing():
    st.sidebar.header("Admin Dashboard")
    go_to = st.sidebar.selectbox('Admin Jobs', ["Manage Users", "Manage Projects", "Annotator View"])


    if go_to == "Manage Users":
        st.header("User Management Dashboard")
        selection = st.sidebar.selectbox("Manage Users", ["", "Register User", "Change Password", "Delete User", "View Users"])
        if selection == "Register User":
            st.subheader("Register User")
            with st.form(key="register_user", clear_on_submit=True):
                data = {}
                data["name"] = st.text_input("Name")
                data["username"] = st.text_input("Username")
                data["password"] = make_hashes(st.text_input("Password", type="password"))
                data["user_type"] = st.selectbox("Role", ["Admin", "Annotator"]).upper()
                data["created_at"] = datetime.datetime.utcnow()
                data["modified_at"] = datetime.datetime.utcnow()
                
                submit_button = st.form_submit_button(label="Register")

            if submit_button:
                fetched_data = Users.objects(username=data["username"])
                if fetched_data:
                    fetched_data = fetched_data[0]

                    st.write("User already exists. Here are the details:")
                    details = dict(fetched_data.to_mongo())
                    details.pop("_id")
                    details.pop("password")
                    st.write(pd.DataFrame.from_records([details]))
                    
                    
                else:
                    success = Users(**data).save()
                    st.success("User Registered")

                
        elif selection == "Change Password":
            st.subheader("Change Password")

            with st.form(key="change_password", clear_on_submit=True):
                username = st.text_input("Username")
                new_password = st.text_input("New Password", type="password")
                confirm_new_password = st.text_input("Confirm New Password", type="password")

                submit_button = st.form_submit_button(label="Change Password")
            
            if submit_button:
                if new_password == confirm_new_password:
                    if update_password(username, new_password):
                        st.success("Password Updated Successfully")
                    else:
                        st.warning("Account is not registered yet!")
                else:
                    st.warning("Confirm password don't match, please check")
                
        elif selection == "Delete User":
            
            st.subheader("Delete User")

            with st.form(key="Delete User", clear_on_submit=True):
                
                username = st.text_input("Username")

                submit_button = st.form_submit_button(label="Delete User")
                
                if submit_button:
                    success = delete_account(username)
                    if success:
                        st.success("Account deleted")
                    else:
                        st.error("Error while deleting account")
                
                        

            if st.session_state.get("confirm_delete_user"):
                confirm_delete  = st.button("Confirm Delete?")
                print("confirm delete", confirm_delete)
                if confirm_delete:
                    st.success("User Deleted successfully")
                else:
                    st.warning("User deletion aborted!")

        elif selection == "View Users":
            st.subheader("Registerd Users List")
            fetched_data = Users.objects().only("name", "username","user_type" ,"deactivated")
            data = []
            if fetched_data:
                for d in fetched_data:
                    d = dict(d.to_mongo())
                    d.pop("_id")
                    data.append(d)
                
                st.dataframe(pd.DataFrame.from_records(data))
            else:
                
                st.success("No Registered Users")
        else:
            st.write("\n\n")
            st.markdown("Please select the related user management tasks from under the `Manage User` dropbox in the panel")
    
    elif go_to == "Manage Projects":
        manage_projects = st.sidebar.selectbox("Project Tasks", ["", "Create Projects", "Update Projects", "List Projects"])
        st.header("Project Management Dashboard")
        if manage_projects == "Create Projects":
            st.subheader("Create Project")
            with st.form(key="create_project", clear_on_submit=True):
                data = {}
                data["project_name"] = st.text_input("Project Name")
                data["project_id"] = st.text_input("Project Id")
                data["project_description"] = st.text_area("Project Description", height=30)
                data["tagging_instructions"] = st.text_area("Tagging Instructions", height=30)
                data["start_date"] = st.date_input("Start Date")
                data["end_date"] = st.date_input("End Date")
                data["tagging_status"] = st.selectbox("Tagging Status", ["Not Started", "In Progress", "Completed", "Archieved"])
                mapping = {"Not Started" : "NOT_STARTED", "In Progress": "IN_PROGRESS", "Completed": "COMPLETED", "Archieved": "ARCHIEVED"}
                data["tagging_status"] = mapping[data["tagging_status"]]
                data["annotators"] = st_tags( 
                    label="Add Users",
                    text="enter email for suggestion and press enter to confirm",
                    value=[],
                    suggestions=fetch_annotators("ANNOTATORS"),
                )
                data["created_at"] = datetime.datetime.utcnow()
                data["modified_at"] = datetime.datetime.utcnow()

                submit_button = st.form_submit_button(label="Create Project")

            if submit_button:
                fetched_data = Projects.objects(project_id=data["project_id"])
                if fetched_data:
                    fetched_data = fetched_data[0]

                    st.write("Project already exists. Here are the details:")
                    details = dict(fetched_data.to_mongo())
                    details.pop("_id")
                    st.write(pd.DataFrame.from_records([details]))

                else:
                    success = Projects(**data).save()
                    st.success("Project Created Successfully")
                
        elif manage_projects == "Update Projects":
            st.subheader("Update Project")
        elif manage_projects == "List Projects":
            st.subheader("Project List")
            fetched_data = Projects.objects().only("project_name", "project_description", "active","tagging_status" ,"start_date", "end_date")
            data = []
            if fetched_data:
                for d in fetched_data:
                    d = dict(d.to_mongo())
                    for a in ["_id", "created_at", "modified_at", "annotators"]:
                        d.pop(a)
                    
                    data.append(d)
                
                
                mapping = {"Not Started" : "NOT_STARTED", "In Progress": "IN_PROGRESS", "Completed": "COMPLETED", "Archieved": "ARCHIEVED"}
                rev_mapping = {v: k for k,v in mapping.items()}
                df = pd.DataFrame.from_records(data)
                df["tagging_status"] = df["tagging_status"].apply(lambda x: rev_mapping[x])
                df["active"] = df["active"].apply(lambda x: "Yes" if x else "No")
                st.markdown(df.to_html(), unsafe_allow_html=True)
            
            else:
                
                st.success("No Registered Projects")

        else:
            st.write("\n\n")
            st.markdown("Please select the related projects tasks from under the `Project Tasks` dropbox in the panel")

    elif go_to == "Annotator View":
        annotator_landing()

