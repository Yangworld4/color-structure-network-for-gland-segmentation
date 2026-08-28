import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int64)
        self.birth = np.zeros(size, dtype=np.float32)
        self.birth_coord = np.zeros((size, 2), dtype=np.int32)
        self.active = np.zeros(size, dtype=bool)

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root


def compute_h0_pairs_superlevel(probability):
    """Finite H0 pairs for foreground superlevel filtration.

    We run an equivalent sublevel filtration on f = 1 - foreground_probability.
    Reported birth/death values are converted back to foreground probabilities,
    so persistence = birth_probability - death_probability. The death
    coordinate is the pixel where two active components merge. 8-connectivity
    matches the 3x3 connected-component convention used elsewhere.
    """
    prob = np.asarray(probability, dtype=np.float32)
    height, width = prob.shape
    filt = 1.0 - prob
    order = np.lexsort((
        np.arange(height * width, dtype=np.int64),
        filt.reshape(-1),
    ))
    uf = UnionFind(height * width)
    pairs = []
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    for flat in order:
        y, x = divmod(int(flat), width)
        value = float(filt[y, x])
        uf.active[flat] = True
        uf.parent[flat] = flat
        uf.birth[flat] = value
        uf.birth_coord[flat] = (y, x)

        roots = set()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                neighbor = ny * width + nx
                if uf.active[neighbor]:
                    roots.add(uf.find(neighbor))
        if not roots:
            continue
        roots.add(uf.find(flat))
        survivor = min(
            roots,
            key=lambda root: (
                float(uf.birth[root]),
                int(uf.birth_coord[root][0]),
                int(uf.birth_coord[root][1]),
            ),
        )
        for root in sorted(roots):
            root = uf.find(root)
            survivor = uf.find(survivor)
            if root == survivor:
                continue
            dead_root = root
            if (
                float(uf.birth[root]),
                int(uf.birth_coord[root][0]),
                int(uf.birth_coord[root][1]),
            ) < (
                float(uf.birth[survivor]),
                int(uf.birth_coord[survivor][0]),
                int(uf.birth_coord[survivor][1]),
            ):
                dead_root, survivor = survivor, root
            birth_f = float(uf.birth[dead_root])
            death_f = value
            by, bx = (
                int(uf.birth_coord[dead_root][0]),
                int(uf.birth_coord[dead_root][1]),
            )
            pairs.append({
                'birth': float(1.0 - birth_f),
                'death': float(1.0 - death_f),
                'persistence': float(death_f - birth_f),
                'birth_coord': (by, bx),
                'death_coord': (y, x),
            })
            uf.parent[dead_root] = survivor
    return pairs, 1


def mark_h0_death_components_superlevel(probability, component_labels, candidates):
    """Return candidate component IDs containing an H0 death critical point.

    This follows the same filtration and tie-breaking as
    ``compute_h0_pairs_superlevel`` but avoids materializing one Python dict per
    pixel. PH Mask Only only needs to know whether a death coordinate falls
    inside a candidate disagreement component.
    """
    prob = np.asarray(probability, dtype=np.float32)
    labels = np.asarray(component_labels)
    height, width = prob.shape
    filt = 1.0 - prob
    order = np.lexsort((
        np.arange(height * width, dtype=np.int64),
        filt.reshape(-1),
    ))
    if njit is not None:
        candidate_flags = np.zeros(int(labels.max()) + 1, dtype=np.bool_)
        target_count = 0
        for component_id in candidates:
            component_id = int(component_id)
            if 0 < component_id < candidate_flags.shape[0]:
                if not candidate_flags[component_id]:
                    target_count += 1
                candidate_flags[component_id] = True
        found_flags = _mark_h0_death_components_numba(
            filt,
            labels.astype(np.int64, copy=False),
            order,
            candidate_flags,
            target_count,
        )
        return set(np.flatnonzero(found_flags))
    uf = UnionFind(height * width)
    found = set()
    candidates = set(int(component_id) for component_id in candidates)
    if not candidates:
        return found
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    for flat in order:
        y, x = divmod(int(flat), width)
        value = float(filt[y, x])
        uf.active[flat] = True
        uf.parent[flat] = flat
        uf.birth[flat] = value
        uf.birth_coord[flat] = (y, x)

        roots = set()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                neighbor = ny * width + nx
                if uf.active[neighbor]:
                    roots.add(uf.find(neighbor))
        if not roots:
            continue
        roots.add(uf.find(flat))
        survivor = min(
            roots,
            key=lambda root: (
                float(uf.birth[root]),
                int(uf.birth_coord[root][0]),
                int(uf.birth_coord[root][1]),
            ),
        )
        for root in sorted(roots):
            root = uf.find(root)
            survivor = uf.find(survivor)
            if root == survivor:
                continue
            dead_root = root
            if (
                float(uf.birth[root]),
                int(uf.birth_coord[root][0]),
                int(uf.birth_coord[root][1]),
            ) < (
                float(uf.birth[survivor]),
                int(uf.birth_coord[survivor][0]),
                int(uf.birth_coord[survivor][1]),
            ):
                dead_root, survivor = survivor, root
            component_id = int(labels[y, x])
            if component_id in candidates:
                found.add(component_id)
                if len(found) == len(candidates):
                    return found
            uf.parent[dead_root] = survivor
    return found


def collect_h0_pairs_for_components_superlevel(probability, component_labels, candidates):
    """Collect finite H0 pairs whose death point falls in candidate components."""
    prob = np.asarray(probability, dtype=np.float32)
    labels = np.asarray(component_labels)
    height, width = prob.shape
    filt = 1.0 - prob
    order = np.lexsort((
        np.arange(height * width, dtype=np.int64),
        filt.reshape(-1),
    ))
    candidate_flags = np.zeros(int(labels.max()) + 1, dtype=np.bool_)
    for component_id in candidates:
        component_id = int(component_id)
        if 0 < component_id < candidate_flags.shape[0]:
            candidate_flags[component_id] = True
    if not candidate_flags.any():
        return []
    if njit is not None:
        arrays = _collect_h0_pairs_for_components_numba(
            filt,
            labels.astype(np.int64, copy=False),
            order,
            candidate_flags,
        )
        return _pairs_from_arrays(arrays)
    return _collect_h0_pairs_for_components_python(filt, labels, order, candidate_flags)


def _pairs_from_arrays(arrays):
    births, deaths, persistences, birth_y, birth_x, death_y, death_x, components, count = arrays
    pairs = []
    for index in range(int(count)):
        pairs.append({
            'birth': float(births[index]),
            'death': float(deaths[index]),
            'persistence': float(persistences[index]),
            'birth_coord': (int(birth_y[index]), int(birth_x[index])),
            'death_coord': (int(death_y[index]), int(death_x[index])),
            'component_id': int(components[index]),
        })
    return pairs


def _collect_h0_pairs_for_components_python(filt, labels, order, candidate_flags):
    height, width = filt.shape
    uf = UnionFind(height * width)
    pairs = []
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for flat in order:
        y, x = divmod(int(flat), width)
        value = float(filt[y, x])
        uf.active[flat] = True
        uf.parent[flat] = flat
        uf.birth[flat] = value
        uf.birth_coord[flat] = (y, x)
        roots = set()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                neighbor = ny * width + nx
                if uf.active[neighbor]:
                    roots.add(uf.find(neighbor))
        if not roots:
            continue
        roots.add(uf.find(flat))
        survivor = min(
            roots,
            key=lambda root: (
                float(uf.birth[root]),
                int(uf.birth_coord[root][0]),
                int(uf.birth_coord[root][1]),
            ),
        )
        for root in sorted(roots):
            root = uf.find(root)
            survivor = uf.find(survivor)
            if root == survivor:
                continue
            dead_root = root
            if (
                float(uf.birth[root]),
                int(uf.birth_coord[root][0]),
                int(uf.birth_coord[root][1]),
            ) < (
                float(uf.birth[survivor]),
                int(uf.birth_coord[survivor][0]),
                int(uf.birth_coord[survivor][1]),
            ):
                dead_root, survivor = survivor, root
            component_id = int(labels[y, x])
            if 0 < component_id < candidate_flags.shape[0] and candidate_flags[component_id]:
                birth_f = float(uf.birth[dead_root])
                by, bx = (
                    int(uf.birth_coord[dead_root][0]),
                    int(uf.birth_coord[dead_root][1]),
                )
                pairs.append({
                    'birth': float(1.0 - birth_f),
                    'death': float(1.0 - value),
                    'persistence': float(value - birth_f),
                    'birth_coord': (by, bx),
                    'death_coord': (y, x),
                    'component_id': component_id,
                })
            uf.parent[dead_root] = survivor
    return pairs


if njit is not None:
    @njit(cache=True)
    def _find_numba(parent, item):
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root


    @njit(cache=True)
    def _root_is_less(root_a, root_b, birth, birth_y, birth_x):
        if birth[root_a] < birth[root_b]:
            return True
        if birth[root_a] > birth[root_b]:
            return False
        if birth_y[root_a] < birth_y[root_b]:
            return True
        if birth_y[root_a] > birth_y[root_b]:
            return False
        return birth_x[root_a] < birth_x[root_b]


    @njit(cache=True)
    def _mark_h0_death_components_numba(
        filt,
        labels,
        order,
        candidate_flags,
        target_count,
    ):
        height, width = filt.shape
        size = height * width
        parent = np.arange(size, dtype=np.int64)
        birth = np.zeros(size, dtype=np.float32)
        birth_y = np.zeros(size, dtype=np.int64)
        birth_x = np.zeros(size, dtype=np.int64)
        active = np.zeros(size, dtype=np.bool_)
        found_flags = np.zeros(candidate_flags.shape[0], dtype=np.bool_)
        found_count = 0
        dy_values = np.array((-1, -1, -1, 0, 0, 1, 1, 1), dtype=np.int64)
        dx_values = np.array((-1, 0, 1, -1, 1, -1, 0, 1), dtype=np.int64)

        for order_index in range(order.shape[0]):
            flat = int(order[order_index])
            y = flat // width
            x = flat - y * width
            value = filt[y, x]
            active[flat] = True
            parent[flat] = flat
            birth[flat] = value
            birth_y[flat] = y
            birth_x[flat] = x

            roots = np.empty(9, dtype=np.int64)
            num_roots = 0
            for neighbor_index in range(8):
                ny = y + int(dy_values[neighbor_index])
                nx = x + int(dx_values[neighbor_index])
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor = ny * width + nx
                    if active[neighbor]:
                        root = _find_numba(parent, neighbor)
                        exists = False
                        for root_index in range(num_roots):
                            if roots[root_index] == root:
                                exists = True
                                break
                        if not exists:
                            roots[num_roots] = root
                            num_roots += 1
            if num_roots == 0:
                continue

            flat_root = _find_numba(parent, flat)
            exists = False
            for root_index in range(num_roots):
                if roots[root_index] == flat_root:
                    exists = True
                    break
            if not exists:
                roots[num_roots] = flat_root
                num_roots += 1

            survivor = roots[0]
            for root_index in range(1, num_roots):
                root = roots[root_index]
                if _root_is_less(root, survivor, birth, birth_y, birth_x):
                    survivor = root

            for left in range(num_roots - 1):
                min_index = left
                for right in range(left + 1, num_roots):
                    if roots[right] < roots[min_index]:
                        min_index = right
                if min_index != left:
                    temp = roots[left]
                    roots[left] = roots[min_index]
                    roots[min_index] = temp

            for root_index in range(num_roots):
                root = _find_numba(parent, roots[root_index])
                survivor = _find_numba(parent, survivor)
                if root == survivor:
                    continue
                dead_root = root
                if _root_is_less(root, survivor, birth, birth_y, birth_x):
                    dead_root = survivor
                    survivor = root
                component_id = int(labels[y, x])
                if (
                    0 < component_id < candidate_flags.shape[0]
                    and candidate_flags[component_id]
                    and not found_flags[component_id]
                ):
                    found_flags[component_id] = True
                    found_count += 1
                    if found_count == target_count:
                        return found_flags
                parent[dead_root] = survivor
        return found_flags


    @njit(cache=True)
    def _collect_h0_pairs_for_components_numba(
        filt,
        labels,
        order,
        candidate_flags,
    ):
        height, width = filt.shape
        size = height * width
        parent = np.arange(size, dtype=np.int64)
        birth = np.zeros(size, dtype=np.float32)
        birth_y = np.zeros(size, dtype=np.int64)
        birth_x = np.zeros(size, dtype=np.int64)
        active = np.zeros(size, dtype=np.bool_)
        out_birth = np.zeros(size, dtype=np.float32)
        out_death = np.zeros(size, dtype=np.float32)
        out_persistence = np.zeros(size, dtype=np.float32)
        out_birth_y = np.zeros(size, dtype=np.int64)
        out_birth_x = np.zeros(size, dtype=np.int64)
        out_death_y = np.zeros(size, dtype=np.int64)
        out_death_x = np.zeros(size, dtype=np.int64)
        out_component = np.zeros(size, dtype=np.int64)
        out_count = 0
        dy_values = np.array((-1, -1, -1, 0, 0, 1, 1, 1), dtype=np.int64)
        dx_values = np.array((-1, 0, 1, -1, 1, -1, 0, 1), dtype=np.int64)

        for order_index in range(order.shape[0]):
            flat = int(order[order_index])
            y = flat // width
            x = flat - y * width
            value = filt[y, x]
            active[flat] = True
            parent[flat] = flat
            birth[flat] = value
            birth_y[flat] = y
            birth_x[flat] = x

            roots = np.empty(9, dtype=np.int64)
            num_roots = 0
            for neighbor_index in range(8):
                ny = y + int(dy_values[neighbor_index])
                nx = x + int(dx_values[neighbor_index])
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor = ny * width + nx
                    if active[neighbor]:
                        root = _find_numba(parent, neighbor)
                        exists = False
                        for root_index in range(num_roots):
                            if roots[root_index] == root:
                                exists = True
                                break
                        if not exists:
                            roots[num_roots] = root
                            num_roots += 1
            if num_roots == 0:
                continue

            flat_root = _find_numba(parent, flat)
            exists = False
            for root_index in range(num_roots):
                if roots[root_index] == flat_root:
                    exists = True
                    break
            if not exists:
                roots[num_roots] = flat_root
                num_roots += 1

            survivor = roots[0]
            for root_index in range(1, num_roots):
                root = roots[root_index]
                if _root_is_less(root, survivor, birth, birth_y, birth_x):
                    survivor = root

            for left in range(num_roots - 1):
                min_index = left
                for right in range(left + 1, num_roots):
                    if roots[right] < roots[min_index]:
                        min_index = right
                if min_index != left:
                    temp = roots[left]
                    roots[left] = roots[min_index]
                    roots[min_index] = temp

            for root_index in range(num_roots):
                root = _find_numba(parent, roots[root_index])
                survivor = _find_numba(parent, survivor)
                if root == survivor:
                    continue
                dead_root = root
                if _root_is_less(root, survivor, birth, birth_y, birth_x):
                    dead_root = survivor
                    survivor = root
                component_id = int(labels[y, x])
                if (
                    0 < component_id < candidate_flags.shape[0]
                    and candidate_flags[component_id]
                ):
                    birth_f = birth[dead_root]
                    out_birth[out_count] = 1.0 - birth_f
                    out_death[out_count] = 1.0 - value
                    out_persistence[out_count] = value - birth_f
                    out_birth_y[out_count] = birth_y[dead_root]
                    out_birth_x[out_count] = birth_x[dead_root]
                    out_death_y[out_count] = y
                    out_death_x[out_count] = x
                    out_component[out_count] = component_id
                    out_count += 1
                parent[dead_root] = survivor
        return (
            out_birth,
            out_death,
            out_persistence,
            out_birth_y,
            out_birth_x,
            out_death_y,
            out_death_x,
            out_component,
            out_count,
        )
