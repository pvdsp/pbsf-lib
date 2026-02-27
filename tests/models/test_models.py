import unittest

import numpy as np

from pbsf.models import NestedWordSet, PatternGraph, PatternSet, PatternTree
from pbsf.nodes import SlopeSignNode
from pbsf.utils.words.nested_word import NestedWord


def create_test_node(slopes):
    return SlopeSignNode({
        "depth": 0,
        "slopes": np.array(slopes),
        "intercepts": np.array(slopes),
        "breakpoints": []
    })

class TestPatternSet(unittest.TestCase):
    def test_creation(self):
        """Test the creation of a PatternSet instance."""
        model = PatternSet()
        self.assertEqual(len(model.nodes), 0)

    def test_update(self):
        """Test updating a PatternSet instance using a chain."""
        model = PatternSet()
        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([1, 1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([1, 1, 1, 1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]
        present = model.update(chain)
        self.assertFalse(any(present))
        present = model.update(chain)
        self.assertTrue(all(present))

        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([1, 1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([-1, 1, 1, 1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]
        present = model.update(chain)
        self.assertEqual(present, [True, True, False])
        present = model.update(chain)
        self.assertEqual(present, [True, True, True])
        self.assertEqual(len(model.nodes[0]), 1)
        self.assertEqual(len(model.nodes[1]), 1)
        self.assertEqual(len(model.nodes[2]), 2)

        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([-1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([1, 1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([-1, 1, 1, 1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]
        present = model.update(chain)
        self.assertEqual(present, [False, True, True])
        present = model.update(chain)
        self.assertEqual(present, [True, True, True])

    def test_contains(self):
        """Check that we can check if a PatternSet contains a chain of nodes."""
        model = PatternSet()
        chain = [create_test_node([1])]

        self.assertFalse(model.contains(chain))
        vertices = model.update(chain)
        self.assertEqual(vertices, [False])
        self.assertTrue(model.contains(chain))

        chain = [create_test_node([1]),
                 create_test_node([1, -1])]

        self.assertFalse(model.contains(chain))
        vertices = model.update(chain)
        self.assertEqual(vertices, [True, False])
        self.assertTrue(model.contains(chain))

class TestPatternTree(unittest.TestCase):
    def test_creation(self):
        """Test the creation of a PatternTree instance."""
        model = PatternTree()
        self.assertEqual(model.root, 0)
        self.assertEqual(model.graph.vertices[model.root]["node"], "root")
        self.assertEqual(model.graph.vertices[model.root]["depth"], -1)

    def test_update(self):
        """Test _chain_to_vertices through update."""
        model = PatternTree()
        chain = [SlopeSignNode({"depth": 0, "slopes": np.array([1]),
                                "intercepts": np.array([1]), "breakpoints": []})]
        vertices = model.update(chain)
        self.assertEqual(model.graph.outgoing(model.root), {1})
        self.assertEqual(len(model.graph.vertices), 2)
        self.assertEqual(model.graph.vertices[vertices[-1]]["node"], chain[0])
        self.assertEqual(model.graph.vertices[vertices[-1]]["depth"], 0)

        model = PatternTree()
        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([1, 1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([1, 1, 1, 1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]
        self.assertEqual(len(model.graph.vertices), 1)

        # Test the update method with a chain of nodes that is not in the tree at all:
        indices = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 4)
        self.assertEqual(model.graph.vertices[indices[1]]["node"], chain[0])
        self.assertEqual(model.graph.vertices[indices[2]]["node"], chain[1])
        self.assertEqual(model.graph.vertices[indices[3]]["node"], chain[2])

        # Test the update method with a chain of nodes that is fully in the tree:
        _ = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 4)

        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([1, 1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([-1, -1, -1, -1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]

        # Chain is in the tree except for the last node:
        indices = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 5)
        self.assertEqual(model.graph.vertices[indices[1]]["node"], chain[0])
        self.assertEqual(model.graph.vertices[indices[2]]["node"], chain[1])
        self.assertEqual(model.graph.vertices[indices[3]]["node"], chain[2])

        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([-1, -1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([1, 1, 1, 1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]

        # Chain is in the tree except for the second-to-last node:
        indices = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 7)
        self.assertEqual(model.graph.vertices[indices[1]]["node"], chain[0])
        self.assertEqual(model.graph.vertices[indices[2]]["node"], chain[1])
        self.assertEqual(model.graph.vertices[indices[3]]["node"], chain[2])

        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([-1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([1, 1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([1, 1, 1, 1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]

        # Test the update method with a chain of nodes that is fully not in the tree:
        indices = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 10)
        self.assertEqual(model.graph.vertices[indices[1]]["node"], chain[0])
        self.assertEqual(model.graph.vertices[indices[2]]["node"], chain[1])
        self.assertEqual(model.graph.vertices[indices[3]]["node"], chain[2])

    def test_contains(self):
        """Check that we can check if a PatternTree contains a chain of nodes."""
        model = PatternTree()
        chain = [create_test_node([1])]

        self.assertFalse(model.contains(chain))
        vertices = model.update(chain)
        self.assertEqual(vertices, [0, 1])
        self.assertTrue(model.contains(chain))

        chain = [create_test_node([1]),
                 create_test_node([1, -1])]

        self.assertFalse(model.contains(chain))
        vertices = model.update(chain)
        self.assertEqual(vertices, [0, 1, 2])
        self.assertTrue(model.contains(chain))


class TestPatternGraph(unittest.TestCase):
    def test_creation(self):
        """Test the creation of a PatternGraph instance."""
        model = PatternGraph()
        self.assertEqual(len(model.graph.vertices), 0)
        self.assertEqual(model.graph.max_depth, 1)

    def test_update(self):
        """Test the update method of a PatternGraph instance."""
        model = PatternGraph()
        chain = [SlopeSignNode({"depth": 0, "slopes": np.array([1]),
                                "intercepts": np.array([]), "breakpoints": []})]
        vertices = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 1)
        self.assertEqual(model.graph.vertices[vertices[-1]]["node"], chain[0])

        model = PatternGraph()
        self.assertEqual(len(model.graph.vertices), 0)

        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([1, 1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([1, 1, 1, 1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]

        # Test the update method with a chain of nodes that is not in the
        indices = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 3)
        self.assertEqual(model.graph.vertices[indices[0]]["node"], chain[0])
        self.assertEqual(model.graph.vertices[indices[1]]["node"], chain[1])
        self.assertEqual(model.graph.vertices[indices[2]]["node"], chain[2])

        # Test the update method with a chain of nodes that is fully in the tree:
        _ = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 3)

        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([1, 1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([-1, -1, -1, -1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]

        # Chain is in the graph except for the last node:
        indices = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 4)
        self.assertEqual(model.graph.vertices[indices[0]]["node"], chain[0])
        self.assertEqual(model.graph.vertices[indices[1]]["node"], chain[1])
        self.assertEqual(model.graph.vertices[indices[2]]["node"], chain[2])

        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([1]),
                "intercepts": np.array([1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([-1, -1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([1, 1, 1, 1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]

        # Chain is in the graph except for second-to-last node.
        # Last node should match an existing node!
        indices = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 5)
        self.assertEqual(model.graph.vertices[indices[0]]["node"], chain[0])
        self.assertEqual(model.graph.vertices[indices[1]]["node"], chain[1])
        self.assertEqual(model.graph.vertices[indices[2]]["node"], chain[2])

        chain = [
            SlopeSignNode({
                "depth": 0,
                "slopes": np.array([-1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 1,
                "slopes": np.array([1, -1]),
                "intercepts": np.array([1, 1]),
                "breakpoints": []
            }),
            SlopeSignNode({
                "depth": 2,
                "slopes": np.array([1, -1, 1, 1]),
                "intercepts": np.array([1, 1, 1, 1]),
                "breakpoints": []
            })
        ]

        # Test the update method with a chain of nodes that is fully not in the graph:
        indices = model.update(chain)
        self.assertEqual(len(model.graph.vertices), 8)
        self.assertEqual(model.graph.vertices[indices[0]]["node"], chain[0])
        self.assertEqual(model.graph.vertices[indices[1]]["node"], chain[1])
        self.assertEqual(model.graph.vertices[indices[2]]["node"], chain[2])

    def test_contains(self):
        """Check that we can check if a PatternGraph contains a chain of nodes."""
        model = PatternGraph()
        chain = [create_test_node([1])]

        self.assertFalse(model.contains(chain))
        vertices = model.update(chain)
        self.assertEqual(vertices, [0])
        self.assertTrue(model.contains(chain))

        chain = [create_test_node([1]),
                 create_test_node([1, -1])]

        self.assertFalse(model.contains(chain))
        vertices = model.update(chain)
        self.assertEqual(vertices, [0, 1])
        self.assertTrue(model.contains(chain))


class TestNestedWordSet(unittest.TestCase):
    def test_creation(self):
        """Creation of an empty NestedWordSet."""
        model = NestedWordSet()
        self.assertEqual(len(model.patterns.graph.vertices), 0)
        self.assertEqual(len(model.nested_words), 0)
        self.assertEqual(model.context_size, 2)

    def test_creation_custom_context_size(self):
        """Creation of an empty NestedWordSet with custom context size."""
        model = NestedWordSet({"context_size": 5})
        self.assertEqual(len(model.patterns.graph.vertices), 0)
        self.assertEqual(len(model.nested_words), 0)
        self.assertEqual(model.context_size, 5)

        with self.assertRaises(ValueError):
            NestedWordSet({"context_size": 0})
        with self.assertRaises(ValueError):
            NestedWordSet({"context_size": -1})

    def test_update_empty_chain(self):
        """Updating with an empty chain should raise ValueError."""
        model = NestedWordSet({"context_size": 2})
        with self.assertRaises(ValueError):
            model.update([])

    def test_update_short_chains(self):
        """Adding chain of length 1 to an empty NestedWord."""
        model = NestedWordSet({"context_size": 2})
        chain1 = [create_test_node([1.0])]
        model.update(chain1)
        self.assertEqual(len(model.patterns.graph.vertices), 1)
        self.assertEqual(len(model.nested_words), 0)

        chain2 = [create_test_node([-1.0])]
        model.update(chain2)
        self.assertEqual(len(model.patterns.graph.vertices), 2)
        self.assertEqual(len(model.nested_words), 1)

        expected_nw = NestedWord.from_tagged_sequence([0, 1])
        self.assertEqual(list(model.nested_words)[0], expected_nw)

    def test_update_duplicate_short_chains(self):
        """Adding duplicate chain of length 1 to an empty NestedWord."""
        model = NestedWordSet({"context_size": 2})
        chain = [create_test_node([1.0])]
        model.update(chain)
        self.assertEqual(len(model.patterns.graph.vertices), 1)
        self.assertEqual(len(model.nested_words), 0)

        model.update(chain)
        self.assertEqual(len(model.patterns.graph.vertices), 1)
        self.assertEqual(len(model.nested_words), 1)

        expected_nw = NestedWord.from_tagged_sequence([0])
        self.assertEqual(list(model.nested_words)[0], expected_nw)

    def test_update_regular_chains(self):
        """Adding chain of length >1 to an empty NestedWord."""
        model = NestedWordSet({"context_size": 4})

        # First chain
        chain1 = [create_test_node([1.0]),
                  create_test_node([1.0, 1.0]),
                  create_test_node([1.0, 1.0, 1.0]),
                  create_test_node([1.0, 1.0, 1.0, 1.0])]
        model.update(chain1)
        self.assertEqual(len(model.patterns.graph.vertices), 4)
        self.assertEqual(len(model.nested_words), 0)

        # Second chain - mismatch at last node
        chain2 = [create_test_node([1.0]),
                  create_test_node([1.0, 1.0]),
                  create_test_node([1.0, 1.0, 1.0]),
                  create_test_node([-1.0, 1.0, 1.0, 1.0])]
        model.update(chain2)
        self.assertEqual(len(model.patterns.graph.vertices), 5)
        self.assertEqual(len(model.nested_words), 0)

        # Third chain - mismatch at second-to-last node
        chain3 = [create_test_node([1.0]),
                  create_test_node([1.0, 1.0]),
                  create_test_node([-1.0, 1.0, 1.0]),
                  create_test_node([1.0, 1.0, 1.0, 1.0])]
        model.update(chain3)
        self.assertEqual(len(model.patterns.graph.vertices), 6)
        self.assertEqual(len(model.nested_words), 0)

        # Fourth chain - mismatch at second node + triggers NestedWord creation
        chain4 = [create_test_node([1.0]),
                  create_test_node([-1.0, 1.0]),
                  create_test_node([1.0, 1.0, 1.0]),
                  create_test_node([1.0, 1.0, 1.0, 1.0])]
        model.update(chain4)
        self.assertEqual(len(model.nested_words), 1)

    def test_update_duplicate_regular_chain(self):
        """Adding duplicate chain of >1 to an empty NestedWord."""
        model = NestedWordSet({"context_size": 2})
        chain = [create_test_node([1.0]),
                 create_test_node([1.0, 1.0]),
                 create_test_node([1.0, 1.0, 1.0]),
                 create_test_node([1.0, 1.0, 1.0, 1.0]),
                 create_test_node([1.0, 1.0, 1.0, 1.0, 1.0])]
        model.update(chain)
        self.assertEqual(len(model.patterns.graph.vertices), 5)
        self.assertEqual(len(model.nested_words), 0)

        model.update(chain)
        self.assertEqual(len(model.patterns.graph.vertices), 5)
        self.assertEqual(len(model.nested_words), 1)

        expected_nw = NestedWord.from_tagged_sequence(
            ['<', 0, '<', 1, '<', 2, '<', 3, 4]
        )
        self.assertTrue(expected_nw in model.nested_words)

    def test_learn(self):
        """Adding a list of chains to the NestedWordSet."""
        model = NestedWordSet({"context_size": 5})
        c = create_test_node
        chains = [
            [c([-1.0, 1.0, 1.0]), c([1.0, 1.0]), c([1.0])],
            [c([1.0, -1.0, 1.0]), c([1.0, 1.0]), c([1.0])],
            [c([1.0, 1.0, -1.0]), c([1.0, 1.0]), c([1.0])],
            [c([1.0, 1.0, 1.0]), c([-1.0, 1.0]), c([1.0])],
            [c([1.0, 1.0, 1.0]), c([1.0, 1.0]), c([-1.0])],
        ]
        result = model.learn(chains)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], NestedWord)

    def test_contains(self):
        """Check that we can check if a NestedWordSet contains a chain of nodes."""
        model = NestedWordSet({"context_size": 1})
        chain = [create_test_node([1])]

        self.assertFalse(model.contains([chain]))
        vertices = model.update(chain)
        self.assertEqual(vertices[0], NestedWord.from_tagged_sequence([0]))
        self.assertTrue(model.contains([chain]))

        chain = [create_test_node([1]),
                 create_test_node([1, -1])]

        self.assertFalse(model.contains([chain]))
        vertices = model.update(chain)
        self.assertEqual(vertices[0], NestedWord.from_tagged_sequence(["<", 0, 1]))
        self.assertTrue(model.contains([chain]))

    def test_contains_no_context(self):
        """Check if chain can be found with context_size 1."""
        model = NestedWordSet({"context_size": 1})
        c = create_test_node
        chains = [
            [c([-1.0]), c([1.0, 1.0]), c([1.0, 1.0, 1.0])],
            [c([1.0]), c([-1.0, 1.0]), c([1.0, 1.0, 1.0])],
            [c([1.0]), c([1.0, -1.0]), c([1.0, 1.0, 1.0])],
            [c([1.0]), c([1.0, 1.0]), c([-1.0, 1.0, 1.0])],
            [c([1.0]), c([1.0, 1.0]), c([1.0, -1.0, 1.0])],
        ]
        for chain in chains:
            model.update(chain)
        for chain in chains:
            self.assertTrue(model.contains([chain]))

    def test_contains_context(self):
        """Check if chain can be found with context_size >1."""
        model = NestedWordSet({"context_size": 5})
        c = create_test_node
        chains = [
            [c([-1.0]), c([1.0, 1.0]), c([1.0, 1.0, 1.0])],
            [c([1.0]), c([-1.0, 1.0]), c([1.0, 1.0, 1.0])],
            [c([1.0]), c([1.0, -1.0]), c([1.0, 1.0, 1.0])],
            [c([1.0]), c([1.0, 1.0]), c([-1.0, 1.0, 1.0])],
            [c([1.0]), c([1.0, 1.0]), c([1.0, -1.0, 1.0])],
        ]

        for _ in range(10):
            for chain in chains:
                model.update(chain)

        # Test sequential order chains
        self.assertTrue(model.contains(chains[0:5]))  # 1-2-3-4-5
        self.assertTrue(model.contains(chains[1:5] + [chains[0]]))  # 2-3-4-5-1
        self.assertTrue(model.contains(chains[2:5] + chains[0:2]))  # 3-4-5-1-2
        self.assertTrue(model.contains(chains[3:5] + chains[0:3]))  # 4-5-1-2-3
        self.assertTrue(model.contains([chains[4]] + chains[0:4]))  # 5-1-2-3-4

        # Test shuffled order (should be False)
        # 2-1-5-3-4
        shuffled = [
            chains[1], chains[0], chains[4],
            chains[2], chains[3],
        ]
        self.assertFalse(model.contains(shuffled))

    def test_contains_edge_cases(self):
        """Test contains method with edge cases."""
        model = NestedWordSet({"context_size": 2})

        # Test contains with empty model
        chain = [create_test_node([1.0])]
        self.assertFalse(model.contains([chain, chain]))

        # Test contains with wrong size
        with self.assertRaises(ValueError):
            self.assertFalse(model.contains([]))  # No chains
        with self.assertRaises(ValueError):
            # Single chain, but context_size > 1
            self.assertFalse(model.contains([chain]))
        with self.assertRaises(ValueError):
            # More than context_size chains
            self.assertFalse(
                model.contains([chain, chain, chain])
            )

        # Add some chains to model
        chains = [
            [create_test_node([1.0])],
            [create_test_node([-1.0])]
        ]
        for chain in chains:
            model.update(chain)

        # Test contains with empty chain list in non-empty model
        with self.assertRaises(ValueError):
            self.assertFalse(model.contains([]))  # No chains

        # Test contains with single chain that exists
        self.assertTrue(model.contains(chains))

        # Test contains with single chain that doesn't exist in model yet
        chain = [create_test_node([1.0, -1.0])]
        self.assertFalse(model.contains([chain, chain]))

        # Test contains with mixed existing and non-existing chains
        mixed_chains = [chains[0], chain]
        self.assertFalse(model.contains(mixed_chains))

    def test_context_queue_boundary_conditions(self):
        """Test context queue behavior at boundaries."""
        # Test with context_size = 1
        model = NestedWordSet({"context_size": 1})
        chain1 = [create_test_node([1.0])]
        chain2 = [create_test_node([1.0])]

        # First update should create NestedWord immediately, as context_size is 1
        result = model.update(chain1)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(model.nested_words), 1)

        result = model.update(chain2)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(model.nested_words), 1)

        # Add exactly context_size chains
        model = NestedWordSet({"context_size": 3})
        chains = [
            [create_test_node([1.0])],
            [create_test_node([-1.0])],
            [create_test_node([1.0, 1.0])]
        ]

        for i, chain in enumerate(chains):
            result = model.update(chain)
            if i < 2:
                self.assertEqual(len(result), 0)  # No NestedWord yet
            else:
                self.assertEqual(len(result), 1)  # NestedWord created

        # Add one more chain to test queue overflow
        overflow_chain = [create_test_node([-1.0, -1.0])]
        result = model.update(overflow_chain)
        self.assertEqual(len(result), 1)  # Should create NestedWord
        self.assertEqual(len(model.nested_words), 2)  # Should have 2 total

    def test_invalid_constructor_parameters(self):
        """Test constructor with invalid parameters."""
        # Test with None parameters (should use defaults)
        model = NestedWordSet(None)
        self.assertEqual(model.context_size, 2)

        # Test with empty dict (should use defaults)
        model = NestedWordSet({})
        self.assertEqual(model.context_size, 2)

        # Test with large context_size
        model = NestedWordSet({"context_size": 1000})
        self.assertEqual(model.context_size, 1000)

    def test_nesting_scenarios(self):
        """Test scenarios with multiple chain interactions."""
        model = NestedWordSet({"context_size": 3})

        # Create chains that will result in complex nested word combinations
        c = create_test_node
        chain1 = [c([1.0]), c([1.0, 1.0])]
        chain2 = [c([1.0]), c([-1.0, 1.0])]
        chain3 = [c([-1.0]), c([1.0, 1.0])]
        chain4 = [c([-1.0]), c([-1.0, -1.0])]

        # Add chains progressively
        result = model.update(chain1)
        self.assertEqual(len(result), 0)
        self.assertEqual(len(model.nested_words), 0)

        result = model.update(chain2)
        self.assertEqual(len(result), 0)
        self.assertEqual(len(model.nested_words), 0)

        result = model.update(chain3)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(model.nested_words), 1)
        expected = NestedWord.from_tagged_sequence(
            ["<", 0, 1, 2, 0, ">", "<", 3, 1]
        )
        self.assertEqual(result[0], expected)

        result = model.update(chain4)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(model.nested_words), 2)
        expected = NestedWord.from_tagged_sequence(
            ["<", 0, 2, 0, ">", "<", 3, 1, 4]
        )
        self.assertEqual(result[0], expected)

    def test_learn_with_empty_and_invalid_chains(self):
        """Test learn method with edge cases."""
        model = NestedWordSet({"context_size": 2})

        # Test learn with empty list
        result = model.learn([])
        self.assertEqual(result, [])
        self.assertEqual(len(model.nested_words), 0)

        # Test learn with list containing empty chain
        with self.assertRaises(ValueError):
            model.learn([[], [create_test_node([1.0])]])

    def test_learn_with_duplicate_context(self):
        """Test learn method with duplicate context chains."""
        model = NestedWordSet({"context_size": 2})
        chains = [
            [create_test_node([1.0])],
            [create_test_node([-1.0])],
            [create_test_node([1.0])],
            [create_test_node([-1.0])]
        ]
        result = model.learn(chains)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(model.nested_words), 2)
