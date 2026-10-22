import pandas as pd

from connectors.Connector import Connector


class InteractionDAO:
    @staticmethod
    def get_all():
        """Mọi tương tác (customer, track) kèm hóa đơn và ngày mua."""
        with Connector() as conn:
            return pd.read_sql_query(
                "SELECT i.CustomerId AS customer_id, il.TrackId AS track_id, "
                "i.InvoiceId AS invoice_id, i.InvoiceDate AS invoice_date "
                "FROM InvoiceLine il JOIN Invoice i ON i.InvoiceId = il.InvoiceId "
                "ORDER BY i.CustomerId, i.InvoiceDate, i.InvoiceId",
                conn,
                parse_dates=["invoice_date"],
            )

    @staticmethod
    def get_history(customer_id):
        """Lịch sử mua của một khách, dùng cho bảng trên UI."""
        with Connector() as conn:
            return pd.read_sql_query(
                "SELECT t.Name AS track, al.Title AS album, ar.Name AS artist, "
                "COALESCE(g.Name, '(không rõ)') AS genre, "
                "i.InvoiceDate AS invoice_date "
                "FROM InvoiceLine il "
                "JOIN Invoice i ON i.InvoiceId = il.InvoiceId "
                "JOIN Track t ON t.TrackId = il.TrackId "
                "LEFT JOIN Album al ON al.AlbumId = t.AlbumId "
                "LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId "
                "LEFT JOIN Genre g ON g.GenreId = t.GenreId "
                "WHERE i.CustomerId = ? ORDER BY i.InvoiceDate DESC",
                conn,
                params=(customer_id,),
            )
