from connectors.Connector import Connector
from models.entities import Track

_SELECT = """
SELECT t.TrackId, t.Name, t.AlbumId, COALESCE(al.Title, ''), COALESCE(ar.ArtistId, 0),
       COALESCE(ar.Name, ''), COALESCE(t.GenreId, 0), COALESCE(g.Name, ''),
       t.MediaTypeId, COALESCE(t.Composer, ''), t.Milliseconds, t.UnitPrice
FROM Track t
LEFT JOIN Album al ON al.AlbumId = t.AlbumId
LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
LEFT JOIN Genre g ON g.GenreId = t.GenreId
"""


class TrackDAO:
    @staticmethod
    def get_all():
        with Connector() as conn:
            rows = conn.execute(_SELECT + " ORDER BY t.TrackId").fetchall()
        return [Track(*tuple(r)) for r in rows]

    @staticmethod
    def get_by_ids(track_ids):
        ids = list(track_ids)
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with Connector() as conn:
            rows = conn.execute(_SELECT + f" WHERE t.TrackId IN ({marks})", ids).fetchall()
        return [Track(*tuple(r)) for r in rows]
