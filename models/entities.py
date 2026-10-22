"""Dataclass thuần dữ liệu, không có hành vi."""
from dataclasses import dataclass


@dataclass
class Customer:
    customer_id: int
    name: str
    country: str
    city: str


@dataclass
class Track:
    track_id: int
    name: str
    album_id: int
    album_title: str
    artist_id: int
    artist_name: str
    genre_id: int
    genre_name: str
    media_type_id: int
    composer: str
    milliseconds: int
    unit_price: float


@dataclass
class Album:
    album_id: int
    title: str
    artist_id: int
    artist_name: str


@dataclass
class Recommendation:
    item_id: int
    name: str
    artist_name: str
    score: float
    reason: str
