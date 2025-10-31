import torch


if torch.__version__.startswith("2.8"):
    import sympy
    import torch.fx.experimental.symbolic_shapes
    from torch.fx.experimental.symbolic_shapes import _lru_cache, _SympyT, safe_expand
    from torch.utils._sympy.functions import CleanDiv, FloorDiv, IntTrueDiv, Max, Min, Mod, TruncToInt

    @_lru_cache
    def simplify(self, expr: _SympyT, size_oblivious: bool = False) -> _SympyT:
        """Use known constraints and replacements to simplify the given
        expr."""
        expr = safe_expand(expr)
        expr = self.replace(expr)

        # Simplify max(0/1, x) to x when x >= 0/1. max(1, x) is a commonly introduced
        # expression when creating contiguous strides.
        if not size_oblivious:
            min_max_replacements = {}
            for atom in expr.atoms(Max):  # type: ignore[has-type]
                if len(atom.args) > 2:
                    continue
                a, b = atom.args
                if b == 1 or b == 0:
                    a, b = b, a

                if a == 1 and self._maybe_evaluate_static(sympy.Ge(b, 1)):
                    min_max_replacements[atom] = b
                if a == 0 and self._maybe_evaluate_static(sympy.Ge(b, 0)):
                    min_max_replacements[atom] = b
            if min_max_replacements:
                expr = expr.xreplace(min_max_replacements)

        if size_oblivious and (expr.has(Max) or expr.has(Min)):  # type: ignore[has-type]
            min_max_replacements = {}
            for atom in (*expr.atoms(Max), *expr.atoms(Min)):  # type: ignore[has-type]
                if len(atom.args) > 2:
                    continue
                a, b = atom.args
                if b == 1 or b == 0:
                    a, b = b, a
                if a == 1 or a == 0:
                    vr = self.bound_sympy(b, size_oblivious=True)
                    if vr.lower >= a:
                        min_max_replacements[atom] = b if atom.func is Max else a
                    elif vr.upper <= a:
                        min_max_replacements[atom] = a if atom.func is Max else b
            if min_max_replacements:
                expr = expr.xreplace(min_max_replacements)

        if expr.has(TruncToInt):
            trunc_replacements = {}
            for atom in expr.atoms(TruncToInt):
                if isinstance(atom.args[0], IntTrueDiv):
                    base, divisor = atom.args[0].args
                    if base % divisor == 0:
                        trunc_replacements[atom] = base // divisor
            if trunc_replacements:
                expr = expr.xreplace(trunc_replacements)

        # TODO it would seem that this pass is not necessary given the
        # below replacement of // with /, but for nested FloorDivs
        # the non-recursive replacement doesn't work, and
        # recursive makes it hard to look up divisibility,
        # because existing divisibility info has FloorDiv in it, not /
        # for now just do a separate pass to catch common nested case
        if expr.has(FloorDiv):
            self._update_divisible()
            div_replacements = {}
            for atom in expr.atoms(FloorDiv):
                base, divisor = atom.args
                if isinstance(divisor, FloorDiv):
                    base1, divisor1 = divisor.args
                    if (
                        self.replace(Mod(base, divisor)) in self.divisible
                        and base == base1
                        and self.replace(Mod(base1, divisor1)) in self.divisible
                    ):
                        div_replacements[atom] = divisor1
            if div_replacements:
                expr = expr.xreplace(div_replacements)
                expr = safe_expand(expr)
        if expr.has(FloorDiv):
            div_replacements = {}
            pows = expr.atoms(sympy.Pow)
            rationals = expr.atoms(sympy.Rational).difference(expr.atoms(sympy.Integer))
            for fd in expr.atoms(FloorDiv):
                base, divisor = fd.args
                if self.replace(Mod(base, divisor)) in self.divisible:
                    div_replacements[fd] = CleanDiv(base, divisor)
            if div_replacements:
                new_expr = expr.xreplace(div_replacements)
                new_expr = safe_expand(new_expr)
                new_pows = new_expr.atoms(sympy.Pow)
                new_rationals = new_expr.atoms(sympy.Rational).difference(new_expr.atoms(sympy.Integer))
                # divisions simplified away
                if new_pows.issubset(pows) and new_rationals.issubset(rationals):
                    expr = new_expr
        return expr

    torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify = simplify  # type: ignore[method-assign]
