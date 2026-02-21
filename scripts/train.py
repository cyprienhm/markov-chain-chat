import json
from pathlib import Path

from markovchainbot import (
    MarkovChain,
    read_discord_package_messages,
    save,
)

parent = Path(__file__).parent
root = (parent / "discord-package" / "messages").resolve()

with open(parent / "guilds_list.json") as f:
    guilds_json = json.load(f)
list_guilds = [guild["server_id"] for guild in guilds_json]

message_paths = []
for channel_dir in root.glob("*/"):
    channel_json_path = channel_dir / "channel.json"
    with open(channel_json_path) as f:
        channel_json = json.load(f)
    if "guild" in channel_json and channel_json["guild"]["id"] in list_guilds:
        channel_msgs_path = channel_dir / "messages.json"
        message_paths.append(channel_msgs_path)

all_messages = [
    message
    for path in message_paths
    for message in read_discord_package_messages(path)
]

v1 = MarkovChain()
v1.add_messages(read_discord_package_messages(parent / "dump.json"))

v2 = MarkovChain()
v2.add_messages(all_messages)

save(v1, Path(parent / "v1.json"))
save(v2, Path(parent / "v2.json"))
