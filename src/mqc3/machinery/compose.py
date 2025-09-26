"""Functions to create a machinery representation that includes multiple machinery representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqc3.machinery.result import MachineryMacronodeMeasuredValue, MachineryResult, MachineryShotMeasuredValue

if TYPE_CHECKING:
    from mqc3.graph.compose import MappingInfo


def decompose_composite_machinery_result(
    composite_machinery_result: MachineryResult,
    map_info: MappingInfo,
) -> MachineryResult:
    """Decompose the execution result of the composite machinery representation.

    This function converts the execution result of the composite machinery representation
    back into the execution results of the original machinery representation.

    Args:
        composite_machinery_result (MachineryResult): MachineryResult object of
            the composite machinery representation.
        map_info (MappingInfo): MappingInfo object.

    Returns:
        MachineryResult: Original machinery result.
    """
    small_machinery_result = MachineryResult([])
    for smv in composite_machinery_result:
        small_smvs: dict[int, MachineryShotMeasuredValue] = {}
        for mmv in smv:
            if mmv.index not in map_info.map:
                continue
            i_shot, macro_idx = map_info.map[mmv.index]
            small_mmv = MachineryMacronodeMeasuredValue(
                mmv.m_a,
                mmv.m_b,
                mmv.m_c,
                mmv.m_d,
                index=macro_idx,
            )
            if i_shot not in small_smvs:
                small_smvs[i_shot] = MachineryShotMeasuredValue({})
            small_smvs[i_shot].items[small_mmv.index] = small_mmv
        small_machinery_result.measured_vals.extend(list(small_smvs.values()))
    return small_machinery_result
