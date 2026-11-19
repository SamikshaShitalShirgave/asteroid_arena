RED, BLACK = True, False


class RBNode:
    __slots__ = ("key", "values", "color", "left", "right", "parent")

    def __init__(self, key, value, color, nil):
        self.key = key
        self.values = [value]
        self.color = color
        self.left = nil
        self.right = nil
        self.parent = nil


class RBTree:
    def __init__(self):
        # One shared black sentinel replaces every None leaf (and the root's parent).
        self.NIL = RBNode(None, None, BLACK, None)
        self.NIL.left = self.NIL.right = self.NIL.parent = self.NIL
        self.NIL.values = []
        self.root = self.NIL
        self.size = 0
        self.rotations = 0  # counted for the report / Tree Theater

    # ---------- rotations ----------
    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left is not self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self.NIL:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        self.rotations += 1

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right is not self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is self.NIL:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        self.rotations += 1

    # ---------- search ----------
    def _find(self, key):
        node, probes, nil = self.root, 0, self.NIL
        while node is not nil:
            probes += 1
            if key == node.key:
                return node, probes
            node = node.left if key < node.key else node.right
        return nil, probes

    def search(self, key):
        """Returns (values list or None, probes)."""
        node, probes = self._find(key)
        return (None if node is self.NIL else node.values), probes

    # ---------- insert ----------
    def insert(self, key, value=None):
        nil, parent, node, probes = self.NIL, self.NIL, self.root, 0
        while node is not nil:
            probes += 1
            if key == node.key:
                node.values.append(value)
                return probes
            parent = node
            node = node.left if key < node.key else node.right
        z = RBNode(key, value, RED, nil)
        z.parent = parent
        if parent is nil:
            self.root = z
        elif key < parent.key:
            parent.left = z
        else:
            parent.right = z
        self.size += 1
        self._insert_fixup(z)
        return probes

    def _insert_fixup(self, z):
        while z.parent.color == RED:
            gp = z.parent.parent
            if z.parent is gp.left:
                uncle = gp.right
                if uncle.color == RED:            # case 1: recolor and move up
                    z.parent.color = BLACK
                    uncle.color = BLACK
                    gp.color = RED
                    z = gp
                else:
                    if z is z.parent.right:       # case 2: turn into case 3
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = BLACK        # case 3: rotate grandparent
                    z.parent.parent.color = RED
                    self._right_rotate(z.parent.parent)
            else:                                 # mirror image
                uncle = gp.left
                if uncle.color == RED:
                    z.parent.color = BLACK
                    uncle.color = BLACK
                    gp.color = RED
                    z = gp
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._left_rotate(z.parent.parent)
        self.root.color = BLACK

    # ---------- delete ----------
    def _transplant(self, u, v):
        if u.parent is self.NIL:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent  # may write to NIL.parent; CLRS relies on this

    def _minimum(self, node):
        while node.left is not self.NIL:
            node = node.left
        return node

    def delete(self, key):
        """Remove a key and all its values. Returns True if it was present."""
        z, _ = self._find(key)
        if z is self.NIL:
            return False
        y = z
        y_color = y.color
        if z.left is self.NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is self.NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        self.size -= 1
        if y_color == BLACK:
            self._delete_fixup(x)
        self.NIL.parent = self.NIL  # tidy the sentinel
        return True

    def _delete_fixup(self, x):
        while x is not self.root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color == RED:                                   # case 1
                    w.color = BLACK
                    x.parent.color = RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:  # case 2
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:                       # case 3
                        w.left.color = BLACK
                        w.color = RED
                        self._right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color                         # case 4
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:                                                    # mirror image
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._right_rotate(x.parent)
                    x = self.root
        x.color = BLACK

    # ---------- range query ----------
    def range(self, lo, hi):
        """Keys in [lo, hi] in order. Returns (keys, probes)."""
        nil = self.NIL
        out, stack, node, probes = [], [], self.root, 0
        while stack or node is not nil:
            if node is not nil:
                probes += 1
                if node.key >= lo:
                    stack.append(node)
                    node = node.left
                else:
                    node = node.right
            else:
                node = stack.pop()
                if node.key > hi:
                    break
                out.append(node.key)
                node = node.right
        return out, probes

    # ---------- helpers ----------
    def inorder(self):
        nil, stack, node = self.NIL, [], self.root
        while stack or node is not nil:
            while node is not nil:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.key
            node = node.right

    def height(self):
        nil = self.NIL
        if self.root is nil:
            return 0
        h, level = 0, [self.root]
        while level:
            h += 1
            level = [c for n in level for c in (n.left, n.right) if c is not nil]
        return h

    def check_invariants(self):
        """Raise AssertionError if any red-black rule is broken. Returns black height."""
        nil = self.NIL
        assert nil.color == BLACK, "NIL must be black"
        assert self.root.color == BLACK, "root must be black"

        def walk(node, lo, hi):
            if node is nil:
                return 1
            assert (lo is None or node.key > lo) and (hi is None or node.key < hi), "BST order broken"
            if node.color == RED:
                assert node.left.color == BLACK and node.right.color == BLACK, "red node with red child"
            for child in (node.left, node.right):
                if child is not nil:
                    assert child.parent is node, "parent pointer broken"
            lh = walk(node.left, lo, node.key)
            rh = walk(node.right, node.key, hi)
            assert lh == rh, "black heights differ"
            return lh + (1 if node.color == BLACK else 0)

        return walk(self.root, None, None)  # depth <= 2 log n, recursion is safe

    def __len__(self):
        return self.size