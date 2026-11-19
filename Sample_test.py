# import random
# from algorithms.bst import BST

# sorted_tree, random_tree = BST(), BST()
# keys = list(range(1, 2001))
# for k in keys:
#     sorted_tree.insert(k)          # catalog order
# random.shuffle(keys)
# for k in keys:
#     random_tree.insert(k)

# print(sorted_tree.height())        # 2000: a linked list
# print(random_tree.height())        # roughly 20-30
# print(sorted_tree.search(2000))    # (values, 2000 probes)



#-------------------------------------------------------------

#Sample test 2 

from algorithms.rbtree import RBTree
t = RBTree()
for k in range(1, 100_001):        # sorted input: the BST's worst case
    t.insert(k)
t.check_invariants()                # no AssertionError = all five rules hold
print(t.height())                   # at most 2*log2(100001) = 33.2
print(t.search(100_000))            # (values, probes <= 33)
print(t.rotations)                  # how much rebalancing sorted input caused