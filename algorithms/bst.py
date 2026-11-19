class Node:
    __slots__ = ("key", "values", "left", "right")

    def __init__(self, key, value):
        self.key = key
        self.values = [value]
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None
        self.size = 0  # number of distinct keys

    def insert(self, key, value=None):
        """Returns the number of probes used. O(h)."""
        if self.root is None:
            self.root = Node(key, value)
            self.size = 1
            return 0
        node, probes = self.root, 0
        while True:
            probes += 1
            if key == node.key:
                node.values.append(value)
                return probes
            if key < node.key:
                if node.left is None:
                    node.left = Node(key, value)
                    break
                node = node.left
            else:
                if node.right is None:
                    node.right = Node(key, value)
                    break
                node = node.right
        self.size += 1
        return probes

    def search(self, key):
        """Returns (values list or None, probes)."""
        node, probes = self.root, 0
        while node is not None:
            probes += 1
            if key == node.key:
                return node.values, probes
            node = node.left if key < node.key else node.right
        return None, probes

    def delete(self, key):
        """Hibbard deletion. Returns True if the key was present."""
        parent, node = None, self.root
        while node is not None and node.key != key:
            parent = node
            node = node.left if key < node.key else node.right
        if node is None:
            return False
        if node.left is not None and node.right is not None:
            # Two children: copy the in-order successor here, then delete the successor.
            succ_parent, succ = node, node.right
            while succ.left is not None:
                succ_parent, succ = succ, succ.left
            node.key, node.values = succ.key, succ.values
            parent, node = succ_parent, succ
        child = node.left if node.left is not None else node.right
        if parent is None:
            self.root = child
        elif parent.left is node:
            parent.left = child
        else:
            parent.right = child
        self.size -= 1
        return True

    def range(self, lo, hi):
        """Keys in [lo, hi] in sorted order. Returns (keys, probes).
        Skips whole subtrees that cannot contain answers."""
        out, stack, node, probes = [], [], self.root, 0
        while stack or node is not None:
            if node is not None:
                probes += 1
                if node.key >= lo:          # left subtree may hold answers
                    stack.append(node)
                    node = node.left
                else:                        # everything on the left is too small
                    node = node.right
            else:
                node = stack.pop()
                if node.key > hi:
                    break                    # in-order: everything after is larger
                out.append(node.key)
                node = node.right
        return out, probes

    def inorder(self):
        stack, node = [], self.root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.key
            node = node.right

    def height(self):
        """Nodes on the longest root-to-leaf path (iterative, level by level)."""
        if self.root is None:
            return 0
        h, level = 0, [self.root]
        while level:
            h += 1
            level = [c for n in level for c in (n.left, n.right) if c is not None]
        return h

    def __len__(self):
        return self.size