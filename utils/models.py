import datetime

from ..constants import TAGGING_STATUS, USER_TYPES
from mongoengine import (BooleanField, DateTimeField, DictField, Document,
                         EmailField, ListField, StringField)


class Users(Document):
    """
    Document class for Users data
    """

    meta = {"collection": "users"}
    name = StringField()
    username = StringField(required=True)
    password = StringField(required=True)
    user_type = StringField(choices=USER_TYPES)
    deactivated = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.datetime.utcnow())
    modified_at = DateTimeField(default=datetime.datetime.utcnow())

class Projects(Document):
    """
    Document class for the Projects
    """

    meta = {"collection": "projects"}
    project_id = StringField(required=True)
    project_name = StringField(required=True)
    project_description = StringField()
    tagging_instructions = StringField()
    annotators = ListField(default=[])
    active = BooleanField(default=True)
    tagging_status = StringField(choices=TAGGING_STATUS)
    start_date = DateTimeField()
    end_date = DateTimeField()
    created_at = DateTimeField(default=datetime.datetime.utcnow())
    modified_at = DateTimeField(default=datetime.datetime.utcnow())


class SEData(Document): # pylint-disable: too-few-public-methods
    """
    Docoment class for the tagging data
    """
    meta = {"collection": "se_data"}
    wav_file = StringField(required=True)
    tagging_status = StringField(default="untagged")
    text = StringField()
    phones = ListField()
    words = ListField()
    g2p_maping = DictField()
    f0_word = ListField()
    energy_word = ListField()
    duration_word = ListField()
    f0_phone = ListField()
    energy_phone = ListField()
    duration_phone = ListField()
    created_at = DateTimeField()
    tagging_status = StringField()
    tagger = StringField(default="")
    tagged_at = DateTimeField(null=True)
    # reviewed_at = DateTimeField(null=True)










