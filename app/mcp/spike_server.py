"""Read-only office lookup used by the tool-calling spike.

This process is a stand-in for the company server. It answers from a fixed
list and does not touch the network, the disk, or any credential.
"""

from __future__ import annotations

from typing import Literal

from mcp.server import MCPServer

FACTS = {
    "reception_phone": "555-0148",
    "weekday_hours": "09:00-17:30",
    "weekend_hours": "closed",
    "mailing_address": "400 Market Street, Suite 12",
    "visitor_parking": "Lot C",
    "lunch_floor": "2",
    "room_booking_code": "BOOK-17",
    "help_desk_extension": "4400",
    "shipping_cutoff": "15:00",
    "holiday_closure": "December 25",
    "badge_office": "Lobby desk",
    "guest_wifi": "North-Guest",
    "assembly_point": "West courtyard",
    "floor3_printer": "HP-3-East",
    "office_timezone": "America/New_York",
    "expense_due_day": "Friday",
    "transit_stop": "Market Street station",
    "package_room_hours": "10:00-16:00",
    "facilities_email": "facilities@example.com",
    "entry_door": "Door A",
}

Topic = Literal[
    "reception_phone",
    "weekday_hours",
    "weekend_hours",
    "mailing_address",
    "visitor_parking",
    "lunch_floor",
    "room_booking_code",
    "help_desk_extension",
    "shipping_cutoff",
    "holiday_closure",
    "badge_office",
    "guest_wifi",
    "assembly_point",
    "floor3_printer",
    "office_timezone",
    "expense_due_day",
    "transit_stop",
    "package_room_hours",
    "facilities_email",
    "entry_door",
]

TOOL_NAME = "office_lookup"


def build_server() -> MCPServer:
    server = MCPServer("office-spike")

    def office_lookup(topic: Topic) -> str:
        fact = FACTS.get(topic)
        if fact is None:
            raise ValueError("unknown topic")
        return fact

    office_lookup.__doc__ = (
        "Look up one office fact. topic must be one of: " + ", ".join(FACTS)
    )
    server.tool()(office_lookup)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
