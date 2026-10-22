from connectors.Connector import Connector
from models.entities import Customer


class CustomerDAO:
    @staticmethod
    def get_all():
        with Connector() as conn:
            rows = conn.execute(
                "SELECT CustomerId, FirstName || ' ' || LastName AS Name, "
                "COALESCE(Country, ''), COALESCE(City, '') "
                "FROM Customer ORDER BY CustomerId"
            ).fetchall()
        return [Customer(*tuple(r)) for r in rows]

    @staticmethod
    def get_by_id(customer_id):
        with Connector() as conn:
            r = conn.execute(
                "SELECT CustomerId, FirstName || ' ' || LastName AS Name, "
                "COALESCE(Country, ''), COALESCE(City, '') "
                "FROM Customer WHERE CustomerId = ?",
                (customer_id,),
            ).fetchone()
        return Customer(*tuple(r)) if r else None
