"""Minimal systemd control over the host's system D-Bus."""
import os

from jeepney import DBusAddress, new_method_call
from jeepney.io.blocking import open_dbus_connection
from jeepney.wrappers import unwrap_msg

BUS = os.environ.get("DBUS_BUS", "SYSTEM").upper()
_MANAGER = DBusAddress("/org/freedesktop/systemd1", bus_name="org.freedesktop.systemd1",
                       interface="org.freedesktop.systemd1.Manager")


def _manager(conn, method, signature, args):
    return unwrap_msg(conn.send_and_get_reply(new_method_call(_MANAGER, method, signature, args)))


def state(unit):
    """ActiveState of a unit: active, inactive, failed, activating, ..."""
    with open_dbus_connection(bus=BUS) as conn:
        (path,) = _manager(conn, "LoadUnit", "s", (unit,))
        props = DBusAddress(path, bus_name="org.freedesktop.systemd1", interface="org.freedesktop.DBus.Properties")
        (value,) = unwrap_msg(conn.send_and_get_reply(
            new_method_call(props, "Get", "ss", ("org.freedesktop.systemd1.Unit", "ActiveState"))))
        return value[1]


def stop(unit):
    with open_dbus_connection(bus=BUS) as conn:
        _manager(conn, "StopUnit", "ss", (unit, "replace"))


def start(unit):
    with open_dbus_connection(bus=BUS) as conn:
        _manager(conn, "StartUnit", "ss", (unit, "replace"))
