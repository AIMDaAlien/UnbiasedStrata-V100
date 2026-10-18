// Frozen native expert laboratory dispatch, generated from expert-20261011T161207Z/winning-dispatch.json.
// Provenance: 600 timing records per mode, 2400 total across modes 1..4.
// Microbenchmark selection only; no accepted total-response performance claim.
// Row classes: <=56, 57..112, >112 average routed rows per expert.
#pragma once
namespace strata::prefill::fused::native_lab_dispatch {
struct Cell { int phase, format, row_class, mode; };
inline constexpr Cell cells[] = {
    {1, 16, 0, 1}, {1, 16, 1, 0}, {1, 16, 2, 0}, // IQ2_XXS
    {1, 17, 0, 0}, {1, 17, 1, 0}, {1, 17, 2, 0}, // IQ2_XS
    {1, 18, 0, 1}, {1, 18, 1, 0}, {1, 18, 2, 0}, // IQ3_XXS
    {1, 21, 0, 1}, {1, 21, 1, 0}, {1, 21, 2, 0}, // IQ3_S
    {1, 22, 0, 0}, {1, 22, 1, 0}, {1, 22, 2, 0}, // IQ2_S
    {2, 20, 0, 2}, {2, 20, 1, 0}, {2, 20, 2, 0}, // IQ4_NL
    {2, 42, 0, 0}, {2, 42, 1, 0}, {2, 42, 2, 0}, // Q2_0
};
static_assert(sizeof(cells) / sizeof(cells[0]) == 21, "all format/phase/row-class cells frozen");
constexpr int mode(int phase, int format, int row_class) {
    for (const Cell& cell : cells)
        if (cell.phase == phase && cell.format == format && cell.row_class == row_class) return cell.mode;
    return 0; // unmeasured formats, including IQ4_XS, retain control
}
}  // namespace strata::prefill::fused::native_lab_dispatch
