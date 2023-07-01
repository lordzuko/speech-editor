from utils.models import Projects

def fetch_available_projects(email):
    print(email)
    fetched_data = Projects.objects(annotators={"$in": [email]}, active=True).only("project_id", "project_name")
    data = []
    if fetched_data:
        for d in fetched_data:
            d = dict(d.to_mongo())
            d.pop("_id")
            data.append({"project_id": d["project_id"], "project_name": d["project_name"]})
    print(data)
    return data

