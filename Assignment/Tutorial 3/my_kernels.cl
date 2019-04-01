// This method finds the minimum value of the vector and stores in B[0] (first element)
kernel void reduce_min(global const int* A, global int* B, local int* scratch) {
    // Declare local variables of sizes and ids
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Cache all N values from global memory to local memory
	scratch[lid] = A[id];

	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through the local size
	for (int i = 1; i < N; i *= 2)
	{
		// If statement to determine whether the value is accepted
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			// Set the local id of the scratch vector to the minimum value of the index
			scratch[lid] = (scratch[lid] < scratch[lid + i]) ? scratch[lid] : scratch[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// We add results from all local groups to the first element of the array
	// Serial operation! but works for any group size
	// Copy the cache to output array
	if (!lid) {
		atomic_min(&B[0], scratch[lid]);
	}
}

// This method finds the maximum value of the vector and stores in B[0] (first element)
kernel void reduce_max(global const int* A, global int* B, local int* scratch) {
    // Declare local variables of sizes and ids
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Cache all N values from global memory to local memory
	scratch[lid] = A[id];

	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through the local size
	for (int i = 1; i < N; i *= 2) {
		// If statement to determine whether the value is accepted
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			// Set the local id of the scratch vector to the maximum value of the index
			scratch[lid] = (scratch[lid] > scratch[lid + i]) ? scratch[lid] : scratch[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// We add results from all local groups to the first element of the array
	// Serial operation! but works for any group size
	// Copy the cache to output array
	if (!lid) {
		atomic_max(&B[0], scratch[lid]);
	}
}

// This method finds the sum value of the vector and stores in B[0] (first element)
kernel void reduce_sum(global const int* A, global int* B, local int* scratch) {
	// Declare local variables of sizes and ids
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Cache all N values from global memory to local memory
	scratch[lid] = A[id];

	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through the local size
	for (int i = 1; i < N; i *= 2)
	{
		// If statement to determine whether the value is accepted
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			// Increment current local id of scratch by this index of scratch
			scratch[lid] += scratch[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// We add results from all local groups to the first element of the array
	// Serial operation! but works for any group size
	// Copy the cache to output array
	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

// This method finds the variance value of the vector and stores in B
kernel void find_variance(global const int* A, global int* B, int mean, int size) {
	// Local variable to get the global id
	int id = get_global_id(0);

	// If global id is less than size of the input vector without padding
	if (id < size)
	{
		// Current id in B = A current id - mean
		B[id] = A[id] - mean;
		// Wait for all local threads to finish copying from global to local memory
		barrier(CLK_LOCAL_MEM_FENCE);
		// Current id in B = B id times by itself
		B[id] = (B[id] * B[id]);
	}
	// This will return the vector and not the first item of scratch
}

// This method finds the variance sum value of the vector and stores in B[0] (First element)
kernel void find_variance_sum(global const int* A, global int* B, local int* scratch, int multiplier) {
	// Declare local variables of sizes and ids
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Cache all N values from global memory to local memory
	scratch[lid] = A[id];

	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through the local size
	for (int i = 1; i < N; i *= 2)
	{
	   // If statement to determine whether the value is accepted
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			// Increment the current index in scratch by the next
			scratch[lid] += scratch[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Divide the scratch local id by the multiplier squared
	scratch[lid] = scratch[lid] / (multiplier * multiplier);

	// We add results from all local groups to the first element of the array
	// Serial operation! but works for any group size
	// Copy the cache to output array
	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

// This method sorts the input vector by using a bubble sort and returns a sorted vector as B
// Unfortunately i wasn't able to do the bubble sorting with scratch and local id.
kernel void bubbleSort(global int* A, global int* B, local int* scratch) {
	// Declare local variables of sizes and ids
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_global_size(0);
	double temp;

	// Cache all N values from global memory to local memory
	scratch[lid] = A[id];
	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop for the size of N
	for (int i = 0; i < N; i++)
	{
		// Loop through the size of N where j = i + 1
		for (int j = i + 1; j < N; j++) 
		{ 
			//If the current index value is greater than the next
			if (A[i] > A[j]) 
			{ 
				// Swap the values
				temp = A[i];
				A[i] = A[j];
				A[j] = temp;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	// Copy the input vector to B
	B[id] = A[id];
}

// This method sorts the input vector by using a selection sort and returns a sorted vector as B
kernel void selectionSort(global const int* A, global int* B, local int* scratch) {
	// Declare local variables of sizes and ids
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_global_size(0);
	int index = 0;

	// Cache all N values from global memory to local memory
	scratch[lid] = A[id];
	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through the global size
	for (int i = 0; i < N; i++)
	{
		// If statement: If the current inex is less than or = to and index is less than id
		if ((A[i] < scratch[lid]) || (A[i] == scratch[lid] && i < id))
			// Increment the index
			index++;

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// Set the value of the B's index to scratch local id
	B[index] = scratch[lid];
}

// This method sorts the input vector by using a selection sort and returns a sorted vector as B
kernel void selectionSort2(global int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int lid = get_local_id(0);
	int index = 0;

	// Cache all N values from global memory to local memory
	scratch[lid] = A[id];
	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through the global size
	for	(int i = 0; i < N; i++)
	{
		// Create a temp variable or A index value
		int temp = A[i];

		// Determine whether current index is smaller
		bool smaller = (temp < scratch[lid]) || (temp == scratch[lid] && i < id);
		
		// If smaller, increment
		index += (smaller)?1:0;
	}
	// B's position = scratch value
	B[index] = scratch[lid];
}