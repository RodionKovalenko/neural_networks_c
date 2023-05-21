#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=Cygwin-Windows
CND_DLIB_EXT=dll
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/main.o \
	${OBJECTDIR}/network_types/feedforward_network.o \
	${OBJECTDIR}/network_types/recurrent_network.o \
	${OBJECTDIR}/utils/activation.o \
	${OBJECTDIR}/utils/array.o \
	${OBJECTDIR}/utils/clear_memory.o \
	${OBJECTDIR}/utils/math.o \
	${OBJECTDIR}/utils/verbose.o \
	${OBJECTDIR}/utils/weight_initializer.o

# Test Directory
TESTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}/tests

# Test Files
TESTFILES= \
	${TESTDIR}/TestFiles/f1

# Test Object Files
TESTOBJECTFILES= \
	${TESTDIR}/tests/newsimpletest1.o

# C Compiler Flags
CFLAGS=-m64 -lm

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=--64

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/network_c.exe

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/network_c.exe: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.c} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/network_c ${OBJECTFILES} ${LDLIBSOPTIONS} -lm

${OBJECTDIR}/main.o: main.c nbproject/Makefile-${CND_CONF}.mk
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -g -s -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.c

${OBJECTDIR}/network_types/feedforward_network.o: network_types/feedforward_network.c nbproject/Makefile-${CND_CONF}.mk
	${MKDIR} -p ${OBJECTDIR}/network_types
	${RM} "$@.d"
	$(COMPILE.c) -g -s -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/network_types/feedforward_network.o network_types/feedforward_network.c

${OBJECTDIR}/network_types/recurrent_network.o: network_types/recurrent_network.c nbproject/Makefile-${CND_CONF}.mk
	${MKDIR} -p ${OBJECTDIR}/network_types
	${RM} "$@.d"
	$(COMPILE.c) -g -s -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/network_types/recurrent_network.o network_types/recurrent_network.c

${OBJECTDIR}/utils/activation.o: utils/activation.c nbproject/Makefile-${CND_CONF}.mk
	${MKDIR} -p ${OBJECTDIR}/utils
	${RM} "$@.d"
	$(COMPILE.c) -g -s -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/activation.o utils/activation.c

${OBJECTDIR}/utils/array.o: utils/array.c nbproject/Makefile-${CND_CONF}.mk
	${MKDIR} -p ${OBJECTDIR}/utils
	${RM} "$@.d"
	$(COMPILE.c) -g -s -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/array.o utils/array.c

${OBJECTDIR}/utils/clear_memory.o: utils/clear_memory.c nbproject/Makefile-${CND_CONF}.mk
	${MKDIR} -p ${OBJECTDIR}/utils
	${RM} "$@.d"
	$(COMPILE.c) -g -s -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/clear_memory.o utils/clear_memory.c

${OBJECTDIR}/utils/math.o: utils/math.c nbproject/Makefile-${CND_CONF}.mk
	${MKDIR} -p ${OBJECTDIR}/utils
	${RM} "$@.d"
	$(COMPILE.c) -g -s -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/math.o utils/math.c

${OBJECTDIR}/utils/verbose.o: utils/verbose.c nbproject/Makefile-${CND_CONF}.mk
	${MKDIR} -p ${OBJECTDIR}/utils
	${RM} "$@.d"
	$(COMPILE.c) -g -s -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/verbose.o utils/verbose.c

${OBJECTDIR}/utils/weight_initializer.o: utils/weight_initializer.c nbproject/Makefile-${CND_CONF}.mk
	${MKDIR} -p ${OBJECTDIR}/utils
	${RM} "$@.d"
	$(COMPILE.c) -g -s -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/weight_initializer.o utils/weight_initializer.c

# Subprojects
.build-subprojects:

# Build Test Targets
.build-tests-conf: .build-tests-subprojects .build-conf ${TESTFILES}
.build-tests-subprojects:

${TESTDIR}/TestFiles/f1: ${TESTDIR}/tests/newsimpletest1.o ${OBJECTFILES:%.o=%_nomain.o}
	${MKDIR} -p ${TESTDIR}/TestFiles
	${LINK.c} -o ${TESTDIR}/TestFiles/f1 $^ ${LDLIBSOPTIONS}   


${TESTDIR}/tests/newsimpletest1.o: tests/newsimpletest1.c 
	${MKDIR} -p ${TESTDIR}/tests
	${RM} "$@.d"
	$(COMPILE.c) -g -s -I. -std=c99 -MMD -MP -MF "$@.d" -o ${TESTDIR}/tests/newsimpletest1.o tests/newsimpletest1.c


${OBJECTDIR}/main_nomain.o: ${OBJECTDIR}/main.o main.c 
	${MKDIR} -p ${OBJECTDIR}
	@NMOUTPUT=`${NM} ${OBJECTDIR}/main.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.c) -g -s -std=c99 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main_nomain.o main.c;\
	else  \
	    ${CP} ${OBJECTDIR}/main.o ${OBJECTDIR}/main_nomain.o;\
	fi

${OBJECTDIR}/network_types/feedforward_network_nomain.o: ${OBJECTDIR}/network_types/feedforward_network.o network_types/feedforward_network.c 
	${MKDIR} -p ${OBJECTDIR}/network_types
	@NMOUTPUT=`${NM} ${OBJECTDIR}/network_types/feedforward_network.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.c) -g -s -std=c99 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/network_types/feedforward_network_nomain.o network_types/feedforward_network.c;\
	else  \
	    ${CP} ${OBJECTDIR}/network_types/feedforward_network.o ${OBJECTDIR}/network_types/feedforward_network_nomain.o;\
	fi

${OBJECTDIR}/network_types/recurrent_network_nomain.o: ${OBJECTDIR}/network_types/recurrent_network.o network_types/recurrent_network.c 
	${MKDIR} -p ${OBJECTDIR}/network_types
	@NMOUTPUT=`${NM} ${OBJECTDIR}/network_types/recurrent_network.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.c) -g -s -std=c99 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/network_types/recurrent_network_nomain.o network_types/recurrent_network.c;\
	else  \
	    ${CP} ${OBJECTDIR}/network_types/recurrent_network.o ${OBJECTDIR}/network_types/recurrent_network_nomain.o;\
	fi

${OBJECTDIR}/utils/activation_nomain.o: ${OBJECTDIR}/utils/activation.o utils/activation.c 
	${MKDIR} -p ${OBJECTDIR}/utils
	@NMOUTPUT=`${NM} ${OBJECTDIR}/utils/activation.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.c) -g -s -std=c99 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/activation_nomain.o utils/activation.c;\
	else  \
	    ${CP} ${OBJECTDIR}/utils/activation.o ${OBJECTDIR}/utils/activation_nomain.o;\
	fi

${OBJECTDIR}/utils/array_nomain.o: ${OBJECTDIR}/utils/array.o utils/array.c 
	${MKDIR} -p ${OBJECTDIR}/utils
	@NMOUTPUT=`${NM} ${OBJECTDIR}/utils/array.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.c) -g -s -std=c99 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/array_nomain.o utils/array.c;\
	else  \
	    ${CP} ${OBJECTDIR}/utils/array.o ${OBJECTDIR}/utils/array_nomain.o;\
	fi

${OBJECTDIR}/utils/clear_memory_nomain.o: ${OBJECTDIR}/utils/clear_memory.o utils/clear_memory.c 
	${MKDIR} -p ${OBJECTDIR}/utils
	@NMOUTPUT=`${NM} ${OBJECTDIR}/utils/clear_memory.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.c) -g -s -std=c99 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/clear_memory_nomain.o utils/clear_memory.c;\
	else  \
	    ${CP} ${OBJECTDIR}/utils/clear_memory.o ${OBJECTDIR}/utils/clear_memory_nomain.o;\
	fi

${OBJECTDIR}/utils/math_nomain.o: ${OBJECTDIR}/utils/math.o utils/math.c 
	${MKDIR} -p ${OBJECTDIR}/utils
	@NMOUTPUT=`${NM} ${OBJECTDIR}/utils/math.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.c) -g -s -std=c99 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/math_nomain.o utils/math.c;\
	else  \
	    ${CP} ${OBJECTDIR}/utils/math.o ${OBJECTDIR}/utils/math_nomain.o;\
	fi

${OBJECTDIR}/utils/verbose_nomain.o: ${OBJECTDIR}/utils/verbose.o utils/verbose.c 
	${MKDIR} -p ${OBJECTDIR}/utils
	@NMOUTPUT=`${NM} ${OBJECTDIR}/utils/verbose.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.c) -g -s -std=c99 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/verbose_nomain.o utils/verbose.c;\
	else  \
	    ${CP} ${OBJECTDIR}/utils/verbose.o ${OBJECTDIR}/utils/verbose_nomain.o;\
	fi

${OBJECTDIR}/utils/weight_initializer_nomain.o: ${OBJECTDIR}/utils/weight_initializer.o utils/weight_initializer.c 
	${MKDIR} -p ${OBJECTDIR}/utils
	@NMOUTPUT=`${NM} ${OBJECTDIR}/utils/weight_initializer.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.c) -g -s -std=c99 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utils/weight_initializer_nomain.o utils/weight_initializer.c;\
	else  \
	    ${CP} ${OBJECTDIR}/utils/weight_initializer.o ${OBJECTDIR}/utils/weight_initializer_nomain.o;\
	fi

# Run Test Targets
.test-conf:
	@if [ "${TEST}" = "" ]; \
	then  \
	    ${TESTDIR}/TestFiles/f1 || true; \
	else  \
	    ./${TEST} || true; \
	fi

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
