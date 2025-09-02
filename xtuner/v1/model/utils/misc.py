def module_dict_repr(self):
    """Return a custom repr for ModuleList that compresses repeated module
    representations."""

    def _addindent(s_, numSpaces):
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    list_of_reprs = [repr(item) for item in self.values()]
    if len(list_of_reprs) == 0:
        return self._get_name() + "()"

    start_end_indices = [[0, 0]]
    repeated_blocks = [list_of_reprs[0]]
    for i, r in enumerate(list_of_reprs[1:], 1):
        if r == repeated_blocks[-1]:
            start_end_indices[-1][1] += 1
            continue

        start_end_indices.append([i, i])
        repeated_blocks.append(r)

    lines = []
    main_str = self._get_name() + "("
    for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
        local_repr = f"({start_id}): {b}"  # default repr

        if start_id != end_id:
            n = end_id - start_id + 1
            local_repr = f"({start_id}-{end_id}): {n} x {b}"

        local_repr = _addindent(local_repr, 2)
        lines.append(local_repr)

    main_str += "\n  " + "\n  ".join(lines) + "\n"
    main_str += ")"
    return main_str
