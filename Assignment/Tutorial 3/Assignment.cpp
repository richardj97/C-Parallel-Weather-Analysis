// Program by: Richard Jacobs (JAC16629926)
// Adapted from workshop tutorials: https://github.com/gcielniak/OpenCL-Tutorials

// Libraries and defines
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono> 

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

// Global variable to multiply data so we can get a decent decimal places
int multiplier = 10;

#pragma region Print Help
// This method outputs a help menu, to help the user navigate around the program
void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}
#pragma endregion

#pragma region Convert nanosecs to secs
// This method takes nanoseconds as parameter and returns seconds by dividing nanoseconds by 1 million
double convertIntervals(double nanoSecs) {
	return nanoSecs / 1000000000;
}
#pragma endregion

#pragma region Read Text File
// This method, takes a file path as parameter and imports the text file, only taking the 6th column and storing into a vector
std::vector<int> ReadFromFile(string path) {
	// New vector
	std::vector<int> tempDataList;
	// New object locating the file path
	ifstream in(path);

	// If the ifstream fails, could be because of the incorrect file path, outputs a error message
	if (in.fail()) { std::cout << "\nUnable to read text file" << std::endl; }
	// Otherwise if the file path is correct
	else {
		// Create local variables to store the data
		string col1;
		int col2, col3, col4, col5;
		double col6;

		// Loop throuh each line, take the column and store into the local variables above
		while (in >> col1 >> col2 >> col3 >> col4 >> col5 >> col6) {
			// We want to store column 6 only. Add column 6 to vector but multiply it by 10 first
			tempDataList.push_back(col6 * multiplier);
		}
		// Close the file stream
		in.close();
	}

	return tempDataList;
}
#pragma endregion

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	// Select platform, device, list platforms or display help
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	// Try statement, if anything goes wrong within these brackets, output an error at the bottom
	try {

		// Part 2 - host operations
		// 2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		// Create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// 2.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "my_kernels.cl");
		cl::Program program(context, sources);

		// Build and debug the kernel code
        // Try statement to catch any errors.. Catch to display the errors if anything goes wrong with the build 
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		#pragma region Select Data Menu
		// This section gets a user input and loads file data depending on the input
		std::vector<int> A;

		// Local variable
		int dOption;

		// Menu
		std::cout << "\n------------- Select data list to execute -------------\n";
		std::cout << "\n1) temp_lincolnshire.txt\n";
		std::cout << "2) temp_lincolnshire_short.txt\n";
		std::cout << "3) test vector\n";
		std::cout << "\nOption: ";

		// dOption = User input
		cin >> dOption;

		// Switch statement to locate the user input
		switch (dOption)
		{
			case 1:
				A = ReadFromFile("temp_lincolnshire.txt");
				break;
			case 2:
				A = ReadFromFile("temp_lincolnshire_short.txt");
				break;
			case 3:
				A = { 9, 63, 16, 22, 45, 18, 100, 1, 4, 7 };
				break;
			default:
				// If an option was chosen but wasn't on the switch statement... use this data as default
				A = ReadFromFile("temp_lincolnshire_short.txt");
				break;
		}
#pragma endregion

		// Part 3 - memory allocation
		// Local variables
		typedef int mytype;
		int aSize = A.size();

		#pragma region Select local, padding, inputs sizes and elements
		// Option menu to get the size of the local size
		// Local variable
		int lsOption;

		// Menu
		std::cout << "\n------------------ Select local size ------------------\n";
		std::cout << "\nNumber of work items: ";

		// lsOption = user input
		cin >> lsOption;

		// The following part adjusts the length of the input vector so it can be run for a specific workgroup size
        // If the total input length is divisible by the workgroup size
        // This makes the code more efficient
		size_t local_size = lsOption;
		size_t padding_size = A.size() % local_size;

		// If the input vector is not a multiple of the local_size
		// Insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		// This is the number of input elements
		size_t input_elements = A.size();

		// This is the size of the elements in bytes
		size_t input_size = A.size() * sizeof(mytype);

		// This is the number of work item groups
		size_t nr_groups = input_elements / local_size;

		// This is the number of output size
		size_t output_size = input_elements * sizeof(mytype);
#pragma endregion

		#pragma region Min, Max, Avg + results
		// Local variables (Vectors) or 'myType' with the number of input elements
		std::vector<mytype> B(input_elements);
		std::vector<mytype> C(input_elements);
		std::vector<mytype> D(input_elements);
		std::vector<mytype> E(input_elements);
		std::vector<unsigned int> F(input_elements);
		std::vector<mytype> G(input_elements);
		
		// Create a new objectfor each profiling event
		cl::Event prof_eventB;
		cl::Event prof_eventC;
		cl::Event prof_eventD;
		cl::Event prof_eventE;
		cl::Event prof_eventF;
		cl::Event prof_eventG;

		// Create a new device buffer objects.. Read and write (Used to pass and get data from the kernel)
		// CL_MEM_READ_ONLY
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);

		// CL_MEM_READ_WRITE
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_F(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_G(context, CL_MEM_READ_WRITE, output_size);

		// Part 4 - device operations
		// 4.1 copy array A to and initialise other arrays on device memory
		// These fills the buffer with the size of the input and passes a vector
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);

		// These fill the buffer and are the returning buffers after the kernel
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_E, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_F, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_G, 0, 0, output_size);

		// 4.2 Setup and execute all kernels (i.e. device code)
		// These calls the kernel methods and pass parameters
		// The first parameter is the input vector, second is the returning vector and third is the local size 
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_min");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype)));

		cl::Kernel kernel_2 = cl::Kernel(program, "reduce_max");
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_C);
		kernel_2.setArg(2, cl::Local(local_size * sizeof(mytype)));

		cl::Kernel kernel_3 = cl::Kernel(program, "reduce_sum");
		kernel_3.setArg(0, buffer_A);
		kernel_3.setArg(1, buffer_D);
		kernel_3.setArg(2, cl::Local(local_size * sizeof(mytype)));

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventB);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventC);
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventD);

		// 4.3 Copy the result from device to host
		// Create a profile to get the exectuion time in nanoseconds
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		uint64_t profileB = prof_eventB.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventB.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);
		uint64_t profileC = prof_eventC.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventC.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_size, &D[0]);
		uint64_t profileD = prof_eventD.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventD.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		// Local variables - Returning vector devide by the multiplieror size of the input vector without padding
		float minValue  = (float)B[0] / multiplier;
		float maxValue  = (float)C[0] / multiplier;
		float meanValue = (float)D[0] / aSize;

		// This kernel passes the mean value as integer and the size of the input vector without padding
		cl::Kernel kernel_4 = cl::Kernel(program, "find_variance");
		kernel_4.setArg(0, buffer_A);
		kernel_4.setArg(1, buffer_E);
		kernel_4.setArg(2, (int)(meanValue));
		kernel_4.setArg(3, aSize);

		queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventE);
		// We want the results from buffer to be copied to E vector with the size of output_size
		queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, output_size, &E[0]);
		uint64_t profileE = prof_eventE.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventE.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		cl::Kernel kernel_5 = cl::Kernel(program, "find_variance_sum");
		kernel_5.setArg(0, buffer_E);
		kernel_5.setArg(1, buffer_F);
		kernel_5.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		kernel_5.setArg(3, multiplier);

		queue.enqueueNDRangeKernel(kernel_5, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventF);
		queue.enqueueReadBuffer(buffer_F, CL_TRUE, 0, output_size, &F[0]);
		uint64_t profileF = prof_eventF.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventF.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		// Local variables
		// Work out the variables and standard deviation. 
		float variance = (float)F[0] / F.size();
		float standardDev = sqrt(variance);

		// Ouput the results from the local variables
		std::cout << "\n----------------------- Results -----------------------\n" << std::endl;
		std::cout << "Work items: " << local_size << std::endl;
		std::cout << "Work groups: " << nr_groups << std::endl;
		std::cout << "Data executed: " << aSize << std::endl;
		std::cout << "Data executed w/padding: " << A.size() <<std::endl;
		std::cout << "\nMinimum: " << std::fixed << std::setprecision(2) << minValue << std::endl;
		std::cout << "Maximum: " << std::fixed << std::setprecision(2) << maxValue << std::endl;
		std::cout << "Mean/Avg: " << std::fixed << std::setprecision(2) << meanValue / multiplier << std::endl;
		std::cout << "Variance: " << std::fixed << std::setprecision(2) << variance << std::endl;
		std::cout << "Standard Deviation: " << std::fixed << std::setprecision(2) << standardDev << std::endl;
#pragma endregion

		#pragma region Sort + Results
		// This is the sorting area

		// Local variables
		int sOption;
		cl::Kernel kernel_6;

		// Menu
		std::cout << "\n-------------- Select sorting algorithm --------------\n";
		std::cout << "\n1) Bubble Sort [SLOW]\n";
		std::cout << "2) Selection Sort\n";
		std::cout << "3) Selection Sort 2\n";
		std::cout << "\nOption: ";

		// sOption = user input
		cin >> sOption;

		// Switch statement -- Select sort based on user input
		// Output selected sort and set kernel to that specific sorting method
		switch (sOption) {
		case 1:
			std::cout << "\nSorting with: Bubble Sort, please wait...\n";
			kernel_6 = cl::Kernel(program, "bubbleSort");
			break;
		case 2:
			std::cout << "\nSorting with: Selection Sort, please wait...\n";
			kernel_6 = cl::Kernel(program, "selectionSort");
			break;
		case 3:
			std::cout << "\nSorting with: Selection Sort 2, please wait...\n";
			kernel_6 = cl::Kernel(program, "selectionSort2");
			break;
		default:
			std::cout << "\nSorting with default: Selection Sort, please wait...\n";
			kernel_6 = cl::Kernel(program, "selectionSort");
			break;
		}

		kernel_6.setArg(0, buffer_A);
		kernel_6.setArg(1, buffer_G);
		kernel_6.setArg(2, cl::Local(local_size * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_6, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventG);
		queue.enqueueReadBuffer(buffer_G, CL_TRUE, 0, output_size, &G[0]);
		uint64_t profileG = prof_eventG.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventG.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		// Local variables to work out half, quarter and third quarter of the returning vector (G)
		int gHalf = (G.size() / 2);
		int gQrt = (G.size() / 4);
		int gThirdQrt = gHalf + gQrt;

		// Local variables to work out the median, Quartile 1 and 3
		float median = (float)G[gHalf] / multiplier;
		float qt_one = (float)G[gQrt] / multiplier ;
		float qt_three = (float)G[gThirdQrt] / multiplier;

		// Output the results
		std::cout << "\n----------------------- Results -----------------------\n" << std::endl;
		std::cout << "1QT: " << qt_one << std::endl;
		std::cout << "3QT: " << qt_three << std::endl;
		std::cout << "Median value: " << median << std::endl;
#pragma endregion

		#pragma region Profiling
		// This area outputs the profiling information from the each kernel
		std::cout << "\n---------------------- Profiling ----------------------\n" << std::endl;
		std::cout << "Minimum: " << "Execution Intervals (ns): " << profileB << " / (secs): " << std::fixed << std::setprecision(6) << convertIntervals(profileB) << "\n" << GetFullProfilingInfo(prof_eventB, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "\nMaximum: " << "Execution Intervals (ns): " << profileC << " / (secs): " << std::fixed << std::setprecision(6) << convertIntervals(profileC) << "\n" << GetFullProfilingInfo(prof_eventC, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "\nMean: " << "Execution Intervals (ns): " << profileD << " / (secs): " << std::fixed << std::setprecision(6) << convertIntervals(profileD) << "\n" << GetFullProfilingInfo(prof_eventD, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "\nVariance: " << "Execution Intervals (ns): " << profileE << " / (secs): " << std::fixed << std::setprecision(6) << convertIntervals(profileE) << "\n" << GetFullProfilingInfo(prof_eventE, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "\nDeviation: " << "Execution Intervals (ns): " << profileF << " / (secs): " << std::fixed << std::setprecision(6) << convertIntervals(profileF) << "\n" << GetFullProfilingInfo(prof_eventF, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "\nVariance + Deviation Execution Intervals (ns): " << profileE + profileF << " / (secs): " << std::fixed << std::setprecision(6) << convertIntervals(profileE) + convertIntervals(profileF) << std::endl;
		std::cout << "\nMedian: " << "Execution Intervals (ns): " << profileG << " / (secs): " << std::fixed << std::setprecision(6) << convertIntervals(profileG) << "\n" << GetFullProfilingInfo(prof_eventG, ProfilingResolution::PROF_US) << "\n" << std::endl;
		//std::cout << G << std::endl;
#pragma endregion
	}
	catch (cl::Error err) {
		// This catches any errors and outputs the following message:
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	// This pauses the system, to stop finishing of the program (return 0).
	std::system("pause");
	return 0;
}
