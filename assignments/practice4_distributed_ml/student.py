"""Complete this function for the individual M4S5 assignment.
The return value multiplies each rank's MEAN loss before backward().
DDP will then average gradients across the participating ranks.
"""

def loss_multiplier(local_count, global_count, world_size):
    """Return a finite numeric multiplier; explain your derivation in the PDF.

    local_count: samples assigned to this rank in the current global batch
    global_count: samples across all ranks in that same batch
    world_size: participating DDP processes

    Requirements: every sample must have equal influence on the global update;
    the equal-sized-shard case must reduce to the original mean-loss behavior.
    """
    raise NotImplementedError('Implement the sample-count correction in student.py')
