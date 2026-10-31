"""Public awards and edge-simple road trails."""


def longest_road(edges: set[tuple[int, int]], blocked: set[int]) -> int:
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for index, (a, b) in enumerate(sorted(edges)):
        adjacency.setdefault(a, []).append((b, index))
        adjacency.setdefault(b, []).append((a, index))

    def walk(vertex: int, used: int) -> int:
        if used and vertex in blocked:
            return 0
        best = 0
        for other, index in adjacency[vertex]:
            bit = 1 << index
            if not used & bit:
                length = 1 + walk(other, used | bit)
                if length > best:
                    best = length
        return best

    return max((walk(vertex, 0) for vertex in adjacency), default=0)


def award_holder(values: list[int], minimum: int, incumbent: int | None) -> int | None:
    maximum = max(values)
    if maximum < minimum:
        return None
    leaders = [seat for seat, value in enumerate(values) if value == maximum]
    if incumbent in leaders:
        return incumbent
    return leaders[0] if len(leaders) == 1 else None
