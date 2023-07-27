import datetime

from constants import TAGGING_STATUS, USER_TYPES
from mongoengine import (BooleanField, DateTimeField, DictField, Document,
                         EmailField, ListField, StringField, IntField)


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


class Annotation(Document): # pylint-disable: too-few-public-methods
    """
    Docoment class for the tagging data
    """
    meta = {"collection": "annotation"}
    wav_name = StringField(required=True)
    tagging_status = StringField(default="untagged")
    text = StringField()
    unedited = DictField()
    edited = DictField()
    p = ListField()
    w = ListField()
    idxs = ListField()
    i2p = DictField()
    i2w = DictField()
    w2p = DictField()
    save_wav_name = StringField()
    created_at = DateTimeField()
    tagging_status = StringField()
    tagger = StringField(default="")
    tagged_at = DateTimeField(null=True)


class Text(Document): # pylint-disable: too-few-public-methods
    """
    Docoment class for the text data
    """
    meta = {"collection": "text"}
    wav_name = StringField(required=True)
    ref_style = StringField(required=True)
    text = StringField(required=True)
    utt_len = IntField(required=True)
    








