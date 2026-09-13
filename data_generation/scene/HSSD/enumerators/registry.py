from __future__ import annotations

from typing import Dict, List

from schema import StateRecord, TaskProposal
from .class01 import Class01Enumerator
from .class02 import Class02Enumerator
from .class03 import Class03Enumerator
from .class04 import Class04Enumerator
from .class05 import Class05Enumerator
from .class06 import Class06Enumerator
from .class07 import Class07Enumerator
from .class08 import Class08Enumerator
from .class09 import Class09Enumerator
from .class10 import Class10Enumerator


def build_registry():
    return [
        Class01Enumerator(),
        Class02Enumerator(),
        Class03Enumerator(),
        Class04Enumerator(),
        Class05Enumerator(),
        Class06Enumerator(),
        Class07Enumerator(),
        Class08Enumerator(),
        Class09Enumerator(),
        Class10Enumerator(),
    ]


def enumerate_proposals(state: StateRecord) -> List[TaskProposal]:
    proposals: List[TaskProposal] = []
    for enumerator in build_registry():
        proposals.extend(enumerator.enumerate(state))
    return proposals

