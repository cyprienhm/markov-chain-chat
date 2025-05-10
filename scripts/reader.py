import json
import pickle
from pathlib import Path

from markovchainbot.markov_chain import MarkovChain
from markovchainbot.utils import (
    DiscordMessageReader,
    DiscordPackageMessageReader,
)

root = Path("./data/discord-package/messages")

with open("./scripts/guilds_list.json") as f:
    guilds_json = json.load(f)
list_guilds = [c["server_id"] for c in guilds_json]

message_paths = []
for channel_dir in root.glob("*/"):
    channel_json_path = channel_dir / "channel.json"
    with open(channel_json_path) as f:
        channel_json = json.load(f)
    if "guild" in channel_json and channel_json["guild"]["id"] in list_guilds:
        channel_msgs_path = channel_dir / "messages.json"
        message_paths.append(channel_msgs_path)

message_reader = DiscordPackageMessageReader()
all_messages = [
    message for c in message_paths for message in message_reader.get_messages(c)
]

hexparrot_v1 = MarkovChain(message_reader=DiscordMessageReader())
hexparrot_v1.add_file_to_vocabulary(Path("./data/dump_Hex0.json"))
hexparrot_v2 = MarkovChain(message_reader=DiscordPackageMessageReader())
hexparrot_v2.add_messages_to_vocabulary(all_messages)


with open("scripts/hexparrot_v1.pkl", "wb") as f:
    pickle.dump(hexparrot_v1, f)
with open("scripts/hexparrot_v2.pkl", "wb") as f:
    pickle.dump(hexparrot_v2, f)
