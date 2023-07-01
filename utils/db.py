import datetime
import re

from utils import check_hashes, make_hashes
from utils.models import Users
from utils.text import get_random_string


def validate_login(email, password):
    fetched_data = Users.objects(user_email=email)
    if fetched_data:
        fetched_data = fetched_data[0]

        if check_hashes(password, fetched_data["password"]):
            return dict(fetched_data.to_mongo())

    return {}


def update_password(email, new_password=""):

    fetched_data = Users.objects(user_email=email)
    if fetched_data:
        fetched_data = fetched_data[0]

        if not new_password:
            new_password = get_random_string(10)
            print("Random New Password:", new_password)

        success = fetched_data.update(
            set__password=make_hashes(new_password),
            set__modified_at=datetime.datetime.utcnow(),
        )

        # TODO: check for success
        # Trigger email
        return True

    return False


def delete_account(email):
    fetched_data = Users.objects(user_email=email)
    if fetched_data:
        fetched_data = fetched_data[0]

        success = fetched_data.update(
            set__deactivated=True, set__modified_at=datetime.datetime.utcnow()
        )
        return True
    return False


def fetch_annotators(user_type):
    fetched_data = Users.objects(user_type=user_type).only("username")
    data = []
    if fetched_data:
        for d in fetched_data:
            d = dict(d.to_mongo())
            d.pop("_id")
            data.append(d["username"])
    print(data)
    return data
