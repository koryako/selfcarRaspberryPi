# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Users/mac/anaconda/envs/tensorflow2.7/bin/cmake

# The command to remove a file.
RM = /Users/mac/anaconda/envs/tensorflow2.7/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/mac/Desktop/ppp/selfcarRaspberryPi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/mac/Desktop/ppp/selfcarRaspberryPi/build

# Include any dependencies generated for this target.
include cpp/CMakeFiles/server.dir/depend.make

# Include the progress variables for this target.
include cpp/CMakeFiles/server.dir/progress.make

# Include the compile flags for this target's objects.
include cpp/CMakeFiles/server.dir/flags.make

cpp/CMakeFiles/server.dir/server.c.o: cpp/CMakeFiles/server.dir/flags.make
cpp/CMakeFiles/server.dir/server.c.o: ../cpp/server.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mac/Desktop/ppp/selfcarRaspberryPi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object cpp/CMakeFiles/server.dir/server.c.o"
	cd /Users/mac/Desktop/ppp/selfcarRaspberryPi/build/cpp && /Users/mac/anaconda/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/server.dir/server.c.o   -c /Users/mac/Desktop/ppp/selfcarRaspberryPi/cpp/server.c

cpp/CMakeFiles/server.dir/server.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/server.dir/server.c.i"
	cd /Users/mac/Desktop/ppp/selfcarRaspberryPi/build/cpp && /Users/mac/anaconda/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/mac/Desktop/ppp/selfcarRaspberryPi/cpp/server.c > CMakeFiles/server.dir/server.c.i

cpp/CMakeFiles/server.dir/server.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/server.dir/server.c.s"
	cd /Users/mac/Desktop/ppp/selfcarRaspberryPi/build/cpp && /Users/mac/anaconda/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/mac/Desktop/ppp/selfcarRaspberryPi/cpp/server.c -o CMakeFiles/server.dir/server.c.s

cpp/CMakeFiles/server.dir/server.c.o.requires:

.PHONY : cpp/CMakeFiles/server.dir/server.c.o.requires

cpp/CMakeFiles/server.dir/server.c.o.provides: cpp/CMakeFiles/server.dir/server.c.o.requires
	$(MAKE) -f cpp/CMakeFiles/server.dir/build.make cpp/CMakeFiles/server.dir/server.c.o.provides.build
.PHONY : cpp/CMakeFiles/server.dir/server.c.o.provides

cpp/CMakeFiles/server.dir/server.c.o.provides.build: cpp/CMakeFiles/server.dir/server.c.o


# Object files for target server
server_OBJECTS = \
"CMakeFiles/server.dir/server.c.o"

# External object files for target server
server_EXTERNAL_OBJECTS =

libserver.dylib: cpp/CMakeFiles/server.dir/server.c.o
libserver.dylib: cpp/CMakeFiles/server.dir/build.make
libserver.dylib: cpp/CMakeFiles/server.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/mac/Desktop/ppp/selfcarRaspberryPi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared library ../libserver.dylib"
	cd /Users/mac/Desktop/ppp/selfcarRaspberryPi/build/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/server.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpp/CMakeFiles/server.dir/build: libserver.dylib

.PHONY : cpp/CMakeFiles/server.dir/build

cpp/CMakeFiles/server.dir/requires: cpp/CMakeFiles/server.dir/server.c.o.requires

.PHONY : cpp/CMakeFiles/server.dir/requires

cpp/CMakeFiles/server.dir/clean:
	cd /Users/mac/Desktop/ppp/selfcarRaspberryPi/build/cpp && $(CMAKE_COMMAND) -P CMakeFiles/server.dir/cmake_clean.cmake
.PHONY : cpp/CMakeFiles/server.dir/clean

cpp/CMakeFiles/server.dir/depend:
	cd /Users/mac/Desktop/ppp/selfcarRaspberryPi/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mac/Desktop/ppp/selfcarRaspberryPi /Users/mac/Desktop/ppp/selfcarRaspberryPi/cpp /Users/mac/Desktop/ppp/selfcarRaspberryPi/build /Users/mac/Desktop/ppp/selfcarRaspberryPi/build/cpp /Users/mac/Desktop/ppp/selfcarRaspberryPi/build/cpp/CMakeFiles/server.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpp/CMakeFiles/server.dir/depend
