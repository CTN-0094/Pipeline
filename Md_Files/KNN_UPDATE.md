# Pipeline Optimization and Changes

## Summary of Changes
This update optimizes the demographic subsampling pipeline by implementing more efficient handling of the NHW (Non-Hispanic White) participant selection process. The main goal was to reduce the runtime while ensuring that the pipeline remains accurate in selecting the nearest NHW participants. Key changes include:

### 1. **Sequential NHW Participant Selection**
   - **Previous Approach**: NHW participants were selected using the K-Nearest Neighbors (KNN) algorithm. Once KNN ran out of valid nearest neighbors, we previously resorted to random sampling from the remaining pool of NHW participants. This approach caused delays and potential inaccuracies.
   - **Current Approach**: We now utilize a sequential method to select the next nearest NHW participants without randomness. Once KNN runs out of neighbors, we move to the next nearest NHW participant from the precomputed nearest neighbor list. This ensures that no randomness is involved, and the closest NHW participants are consistently selected until we reach the desired count.

### 2. **Precomputation of Neighbor Indices**
   - **Previous Approach**: Each step in the pipeline required recalculating neighbors, adding computational overhead and slowing down the process.
   - **Current Approach**: All nearest neighbors for NHW participants are precomputed using KNN at the start of the pipeline. This precomputation allows the pipeline to efficiently reference the next nearest NHW participants without recalculating them repeatedly, significantly reducing the overhead.

### 3. **Final Subset Handling**
   - **Previous Approach**: Random sampling was used to fill the final subset of 1000 NHW participants if the KNN neighbors were exhausted. This could sometimes introduce inefficiency and inaccuracies due to random selection.
   - **Current Approach**: We continue adding the next closest NHW participants from the precomputed list until the final subset reaches exactly 1000 NHW participants. This ensures both accuracy and consistency while maintaining a predictable runtime.

## Performance Improvement
- **Pipeline Runtime**: These optimizations have effectively halved the runtime of the pipeline. Previously, a pipeline run took approximately 43 seconds. With the new changes, the runtime is reduced to around 20 seconds. This significant improvement in runtime is largely due to the elimination of redundant computations and the sequential NHW selection process.
