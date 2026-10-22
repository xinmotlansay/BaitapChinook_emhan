from connectors.Connector import Connector
from models.entities import Album


class AlbumDAO:
    @staticmethod
    def get_all():
        with Connector() as conn:
            rows = conn.execute(
                "SELECT al.AlbumId, al.Title, ar.ArtistId, ar.Name "
                "FROM Album al JOIN Artist ar ON ar.ArtistId = al.ArtistId "
                "ORDER BY al.AlbumId"
            ).fetchall()
        return [Album(*tuple(r)) for r in rows]
