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
CND_PLATFORM=GNU-Linux
CND_DLIB_EXT=so
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
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cppapplication_1

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cppapplication_1: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.c} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cppapplication_1 ${OBJECTFILES} ${LDLIBSOPTIONS} -lm

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

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
