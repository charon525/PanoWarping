# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/e/CmakeDemo/PanoWarping

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/e/CmakeDemo/PanoWarping/build

# Include any dependencies generated for this target.
include CMakeFiles/SRC.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SRC.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SRC.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SRC.dir/flags.make

CMakeFiles/SRC.dir/src/glad.c.o: CMakeFiles/SRC.dir/flags.make
CMakeFiles/SRC.dir/src/glad.c.o: ../src/glad.c
CMakeFiles/SRC.dir/src/glad.c.o: CMakeFiles/SRC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/CmakeDemo/PanoWarping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/SRC.dir/src/glad.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/SRC.dir/src/glad.c.o -MF CMakeFiles/SRC.dir/src/glad.c.o.d -o CMakeFiles/SRC.dir/src/glad.c.o -c /mnt/e/CmakeDemo/PanoWarping/src/glad.c

CMakeFiles/SRC.dir/src/glad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/SRC.dir/src/glad.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/e/CmakeDemo/PanoWarping/src/glad.c > CMakeFiles/SRC.dir/src/glad.c.i

CMakeFiles/SRC.dir/src/glad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/SRC.dir/src/glad.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/e/CmakeDemo/PanoWarping/src/glad.c -o CMakeFiles/SRC.dir/src/glad.c.s

CMakeFiles/SRC.dir/src/config.cpp.o: CMakeFiles/SRC.dir/flags.make
CMakeFiles/SRC.dir/src/config.cpp.o: ../src/config.cpp
CMakeFiles/SRC.dir/src/config.cpp.o: CMakeFiles/SRC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/CmakeDemo/PanoWarping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/SRC.dir/src/config.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SRC.dir/src/config.cpp.o -MF CMakeFiles/SRC.dir/src/config.cpp.o.d -o CMakeFiles/SRC.dir/src/config.cpp.o -c /mnt/e/CmakeDemo/PanoWarping/src/config.cpp

CMakeFiles/SRC.dir/src/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SRC.dir/src/config.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/CmakeDemo/PanoWarping/src/config.cpp > CMakeFiles/SRC.dir/src/config.cpp.i

CMakeFiles/SRC.dir/src/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SRC.dir/src/config.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/CmakeDemo/PanoWarping/src/config.cpp -o CMakeFiles/SRC.dir/src/config.cpp.s

CMakeFiles/SRC.dir/src/localwarp.cpp.o: CMakeFiles/SRC.dir/flags.make
CMakeFiles/SRC.dir/src/localwarp.cpp.o: ../src/localwarp.cpp
CMakeFiles/SRC.dir/src/localwarp.cpp.o: CMakeFiles/SRC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/CmakeDemo/PanoWarping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/SRC.dir/src/localwarp.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SRC.dir/src/localwarp.cpp.o -MF CMakeFiles/SRC.dir/src/localwarp.cpp.o.d -o CMakeFiles/SRC.dir/src/localwarp.cpp.o -c /mnt/e/CmakeDemo/PanoWarping/src/localwarp.cpp

CMakeFiles/SRC.dir/src/localwarp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SRC.dir/src/localwarp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/CmakeDemo/PanoWarping/src/localwarp.cpp > CMakeFiles/SRC.dir/src/localwarp.cpp.i

CMakeFiles/SRC.dir/src/localwarp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SRC.dir/src/localwarp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/CmakeDemo/PanoWarping/src/localwarp.cpp -o CMakeFiles/SRC.dir/src/localwarp.cpp.s

CMakeFiles/SRC.dir/src/lsd.cpp.o: CMakeFiles/SRC.dir/flags.make
CMakeFiles/SRC.dir/src/lsd.cpp.o: ../src/lsd.cpp
CMakeFiles/SRC.dir/src/lsd.cpp.o: CMakeFiles/SRC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/CmakeDemo/PanoWarping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/SRC.dir/src/lsd.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SRC.dir/src/lsd.cpp.o -MF CMakeFiles/SRC.dir/src/lsd.cpp.o.d -o CMakeFiles/SRC.dir/src/lsd.cpp.o -c /mnt/e/CmakeDemo/PanoWarping/src/lsd.cpp

CMakeFiles/SRC.dir/src/lsd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SRC.dir/src/lsd.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/CmakeDemo/PanoWarping/src/lsd.cpp > CMakeFiles/SRC.dir/src/lsd.cpp.i

CMakeFiles/SRC.dir/src/lsd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SRC.dir/src/lsd.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/CmakeDemo/PanoWarping/src/lsd.cpp -o CMakeFiles/SRC.dir/src/lsd.cpp.s

CMakeFiles/SRC.dir/src/globalwarp.cpp.o: CMakeFiles/SRC.dir/flags.make
CMakeFiles/SRC.dir/src/globalwarp.cpp.o: ../src/globalwarp.cpp
CMakeFiles/SRC.dir/src/globalwarp.cpp.o: CMakeFiles/SRC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/CmakeDemo/PanoWarping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/SRC.dir/src/globalwarp.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SRC.dir/src/globalwarp.cpp.o -MF CMakeFiles/SRC.dir/src/globalwarp.cpp.o.d -o CMakeFiles/SRC.dir/src/globalwarp.cpp.o -c /mnt/e/CmakeDemo/PanoWarping/src/globalwarp.cpp

CMakeFiles/SRC.dir/src/globalwarp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SRC.dir/src/globalwarp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/CmakeDemo/PanoWarping/src/globalwarp.cpp > CMakeFiles/SRC.dir/src/globalwarp.cpp.i

CMakeFiles/SRC.dir/src/globalwarp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SRC.dir/src/globalwarp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/CmakeDemo/PanoWarping/src/globalwarp.cpp -o CMakeFiles/SRC.dir/src/globalwarp.cpp.s

# Object files for target SRC
SRC_OBJECTS = \
"CMakeFiles/SRC.dir/src/glad.c.o" \
"CMakeFiles/SRC.dir/src/config.cpp.o" \
"CMakeFiles/SRC.dir/src/localwarp.cpp.o" \
"CMakeFiles/SRC.dir/src/lsd.cpp.o" \
"CMakeFiles/SRC.dir/src/globalwarp.cpp.o"

# External object files for target SRC
SRC_EXTERNAL_OBJECTS =

libSRC.a: CMakeFiles/SRC.dir/src/glad.c.o
libSRC.a: CMakeFiles/SRC.dir/src/config.cpp.o
libSRC.a: CMakeFiles/SRC.dir/src/localwarp.cpp.o
libSRC.a: CMakeFiles/SRC.dir/src/lsd.cpp.o
libSRC.a: CMakeFiles/SRC.dir/src/globalwarp.cpp.o
libSRC.a: CMakeFiles/SRC.dir/build.make
libSRC.a: CMakeFiles/SRC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/e/CmakeDemo/PanoWarping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library libSRC.a"
	$(CMAKE_COMMAND) -P CMakeFiles/SRC.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SRC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SRC.dir/build: libSRC.a
.PHONY : CMakeFiles/SRC.dir/build

CMakeFiles/SRC.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SRC.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SRC.dir/clean

CMakeFiles/SRC.dir/depend:
	cd /mnt/e/CmakeDemo/PanoWarping/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/e/CmakeDemo/PanoWarping /mnt/e/CmakeDemo/PanoWarping /mnt/e/CmakeDemo/PanoWarping/build /mnt/e/CmakeDemo/PanoWarping/build /mnt/e/CmakeDemo/PanoWarping/build/CMakeFiles/SRC.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SRC.dir/depend

