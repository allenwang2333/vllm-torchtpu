# SPDX-License-Identifier: Apache-2.0
"""原创自然语言 MTP 输入：精确 token 长度，完整保留末尾写作任务。"""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.inputs import TokensPrompt


_SITES = (
    "the willow grove beside the ferry landing",
    "the footpath behind the village school",
    "the meadow below the abandoned mill",
    "the bend opposite the community orchard",
    "the reed beds near the wooden bridge",
    "the gardens along the western embankment",
    "the shallow inlet behind the boat workshop",
    "the common ground beside the market square",
    "the stream joining the river below the library",
    "the floodplain beyond the railway cottages",
    "the fishing steps beside the old warehouse",
    "the riverside pasture below the stone chapel",
)

# 每组把观察、验证与后续行动绑定，避免随机拼出互不相干的因果关系。
_FIELD_NOTES = (
    (
        "Water lingered in the wheel ruts after the rest of the path had dried",
        (
            "We compared the compacted ground with an untouched patch nearby and "
            "watched rain soak into the loose soil much faster"
        ),
        (
            "The group proposed moving heavy deliveries to the upper road and "
            "loosening a small test strip before attempting a larger repair"
        ),
    ),
    (
        "Young trees leaned toward the river where the bank had begun to slip",
        (
            "A careful walk along the edge revealed exposed roots but no fresh "
            "cracks beyond the narrow strip used by visitors"
        ),
        (
            "We agreed to move the walking route inland and observe the bank "
            "through another wet season before deciding whether to reinforce it"
        ),
    ),
    (
        "The reeds sheltered small fish even where the open water looked empty",
        (
            "We returned at different hours and found that shade and slower "
            "currents explained more of the pattern than the depth alone"
        ),
        (
            "The volunteers suggested leaving alternating stretches uncut so "
            "that maintenance would preserve connected areas of shelter"
        ),
    ),
    (
        "Visitors avoided a pleasant stretch because its entrance was hidden",
        (
            "Walking the route with newcomers showed that the uncertainty began "
            "at the junction rather than at the narrow gate itself"
        ),
        (
            "A simple sign and a clearer sight line became our first experiment, "
            "with a review planned before spending money on a replacement gate"
        ),
    ),
    (
        "A layer of silt covered the lower stones after the overnight rain",
        (
            "We followed the muddy water upstream and found a bare patch of "
            "ground draining directly into a roadside channel"
        ),
        (
            "The maintenance team offered to plant that patch and record future "
            "storms before deciding whether a settling pond was also needed"
        ),
    ),
    (
        "Birdsong was strongest near the untidy boundary of shrubs and grass",
        (
            "Our quiet surveys found feeding birds using both the seeds in the "
            "grass and the cover of the neighboring branches"
        ),
        (
            "The gardeners proposed a staggered cutting schedule that would "
            "keep these adjoining habitats available throughout the year"
        ),
    ),
    (
        "An otherwise useful bench became uncomfortable in the afternoon sun",
        (
            "We traced the moving shade and asked older walkers where they "
            "usually stopped rather than choosing a location from the map"
        ),
        (
            "The plan placed a second seat under existing trees while leaving "
            "the sunny bench available during the colder part of the year"
        ),
    ),
    (
        "Discarded packaging collected against the same branches each week",
        (
            "A walk after the market closed showed that the overflowing street "
            "bins contributed more litter than the visitors beside the water"
        ),
        (
            "We asked the market stewards to adjust collection times and agreed "
            "to compare the next several cleanups before installing more bins"
        ),
    ),
    (
        "A familiar wildflower had disappeared from the closely mown verge",
        (
            "An older photograph and a surviving patch behind the fence "
            "suggested that mowing now happened before the seeds could mature"
        ),
        (
            "The groundskeeper proposed delaying the first cut on a marked "
            "section while keeping the edge beside the path short and accessible"
        ),
    ),
    (
        "The new drainage outlet made a surprising noise during heavy showers",
        (
            "Listening from nearby homes revealed that the empty pipe carried "
            "sound farther than anyone had expected during the dry inspection"
        ),
        (
            "The engineers planned a small adjustment to the outlet and invited "
            "the residents to judge the result during the next period of rain"
        ),
    ),
)

_WITNESSES = (
    ("Mara, the librarian", "how these observations would enter the public archive"),
    ("Jonas, a boat builder", "whether the proposal could be maintained locally"),
    ("Leila, a schoolteacher", "what the children could safely observe themselves"),
    ("Ravi, a gardener", "how the work would change with the seasons"),
    ("Elena, a retired nurse", "whether the route would remain welcoming to everyone"),
    ("Tomas, a shopkeeper", "how to explain the disruption to regular visitors"),
    ("Ada, a field ecologist", "which observations might challenge our first guess"),
    ("Noor, a local reporter", "whose experience was still missing from the account"),
)

_PERSPECTIVES = (
    (
        (
            "These are field notes for a book about a town restoring its river. "
            "The narrator follows the practical work and the people doing it."
        ),
        (
            "a reflective nonfiction narrative about the restoration, connecting "
            "specific observations with the town's decisions"
        ),
    ),
    (
        (
            "These are a reporter's notes about a community learning to care for "
            "its river. The book follows conversations, setbacks, and gradual change."
        ),
        (
            "a narrative essay from the reporter's perspective, developing the "
            "people, disagreements, and consequences described in the notes"
        ),
    ),
)


def build_mtp_prompts(tokenizer, prompt_tokens: int) -> list[TokensPrompt]:
    """返回两篇恰好含 prompt_tokens 个 token 的原创自然语言输入。

    使用 add_special_tokens=False，并直接交付 token IDs，避免解码再编码
    改变长度。正文最后一段可在 token 边界截断；完整写作指令始终在末尾。
    有限语料由不同地点、完整调查事件和人物视角组合，段落不重复；不足以
    满足预算时显式失败，不循环复制语料。当前用于 512 和 65536 长度案例。
    """
    prompts = []
    for seed, (introduction, writing_task) in enumerate(_PERSPECTIVES):
        opening = tokenizer.encode(introduction + "\n\n", add_special_tokens=False)
        ending = tokenizer.encode(
            "\n\nWrite the next chapter as "
            + writing_task
            + ". Use sustained natural prose with concrete scenes and connected "
            "paragraphs. Develop the account at length rather than listing or "
            "summarizing the notes. Begin the chapter now.\n",
            add_special_tokens=False,
        )
        body_budget = prompt_tokens - len(opening) - len(ending)
        if body_budget <= 0:
            raise ValueError(
                f"prompt_tokens={prompt_tokens} leaves no room for field notes "
                f"after {len(opening) + len(ending)} framing tokens"
            )

        records = list(itertools.product(_SITES, _FIELD_NOTES, _WITNESSES))
        random.Random(seed).shuffle(records)
        body = []
        for site, (observation, investigation, action), (name, concern) in records:
            paragraph = (
                f"Our visit to {site} began with a conversation. {observation}. "
                f"{investigation}. {action}. {name} asked us to consider "
                f"{concern}.\n\n"
            )
            body.extend(tokenizer.encode(paragraph, add_special_tokens=False))
            if len(body) >= body_budget:
                break
        if len(body) < body_budget:
            raise ValueError(
                f"prompt_tokens={prompt_tokens} exceeds the original field-note corpus"
            )
        prompts.append({"prompt_token_ids": opening + body[:body_budget] + ending})
    return prompts
