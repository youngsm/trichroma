#!/bin/bash

# chroma-setup.sh
#
# Script to set up chroma, which automates steps 3-7 in the install guide;
# see http://chroma.bitbucket.org/install.html
#
# A. Mastbaum (mastbaum@hep.upenn.edu), 1/2012
#

CUDA_PATH=/usr/local/cuda/bin
CUDA_LIB=/usr/local/cuda/lib64
ROOT_CONFIG_OPTS="--enable-minuit2 --enable-roofit"

ROOT_VER="5.32.02"
ROOT_TAR="root_v${ROOT_VER}.source.tar.gz"
ROOT_URL="ftp://root.cern.ch/root/${ROOT_TAR}"

GEANT4_VER="4.9.5.p01"
GEANT4_TAR="geant${GEANT4_VER}.tar.gz"
GEANT4_URL="http://geant4.cern.ch/support/source/${GEANT4_TAR}"

G4PY_REPOSITORY="https://bitbucket.org/seibert/g4py#geant4.9.5.p01"
CHROMA_URL="hg+http://bitbucket.org/chroma/chroma#egg=Chroma"

### you should not need to edit below this line ###

## handle command-line options
ncpus=1
envname="chroma_env"
gcc_suffix=""
usage="$(basename $0) [-h] [-i] [-g gcc-suffix] [-j ncpus] [-n env_name] -- set up chroma

  -h: show this help message
  -g: Suffix on gcc binary names to use, defaults to none (Ex: 4.4)
  -j: number of cpus to use in build (default: 1)
  -n: set the name of the virtual environment (default: chroma_env)"

while getopts ':hij:n:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    g) gcc_suffix=$OPTARG
       ;;
    j) ncpus=$OPTARG
       ;;
    n) envname=$OPTARG
       ;;
    ?) printf "illegal option: '%s'\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done

## step 3: virtualenv
# virtualenv
echo "setting up chroma in virtualenv $envname... "
if dpkg --compare-versions `virtualenv --version` '>=' 1.7
then
  VIRTUALENV_FLAGS="--system-site-packages"
else
  VIRTUALENV_FLAGS=""
fi

if virtualenv $VIRTUALENV_FLAGS $envname
then echo "OK"
else
  echo "failed to setup virtualenv"
  exit 1
fi

# symlinks
if [ -n "$gcc_suffix" ]
then
  printf "setting up compiler symlinks... "
  for l in gcc g++ cpp gcov gfortran
  do
    ln -s /usr/bin/$l-$GCC_VERSION $envname/bin/$l
    if [ ! -h $l ]
      then echo "error creating symbolic link $l"
      exit 1
    fi
  done
  echo "OK"
fi

# paths in activate
printf "setting paths in bin/activate... "
if ( echo -e "\nexport PATH=$CUDA_PATH:\$PATH\nexport LD_LIBRARY_PATH=$CUDA_LIB:\$VIRTUAL_ENV/lib:\$LD_LIBRARY_PATH\n" >> $envname/bin/activate )
then echo "OK"
else
  echo "error in write"
  exit 1
fi

# source it!
source $envname/bin/activate
mkdir $VIRTUAL_ENV/src/

## step 4: root
echo "setting up root... "
# fetch, untar and relocate
cd $VIRTUAL_ENV/src
if ! wget --continue $ROOT_URL
then
  echo "error downloading root"
  exit 1
fi

if ! tar xf $ROOT_TAR
then
  echo "error unpacking root"
  exit 1
fi
mv root $VIRTUAL_ENV/src/root-${ROOT_VER}
cd $VIRTUAL_ENV/src/root-${ROOT_VER}

# configure
if ! ./configure $ROOT_CONFIG_OPTS
then
  echo "error configuring root"
  exit 1
fi

# make
if ! make -j$ncpus
then
  echo "error building root"
  exit 1
fi

# update activate script
echo -e "source \$VIRTUAL_ENV/src/root-${ROOT_VER}/bin/thisroot.sh\n" >> $VIRTUAL_ENV/bin/activate

echo "OK"

## step 5: geant4

# geant4 
echo "setting up geant4... "
# fetch, untar and relocate
cd $VIRTUAL_ENV/src
if ! wget --continue $GEANT4_URL
then
  echo "error downloading root"
  exit 1
fi

if ! tar xf $GEANT4_TAR
then
  echo "error unpacking root"
  exit 1
fi

# build geant4 with cmake
cd $VIRTUAL_ENV/src/
mkdir geant${GEANT4_VER}-build
cd geant${GEANT4_VER}-build
if ! cmake -DGEANT4_INSTALL_DATA=ON -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV ../geant${GEANT4_VER}
then
  echo "error building geant4"
  exit 1
fi
if ! make -j$ncpus install
then
  echo "error installing geant4"
  exit 1
fi

# update activate script
echo -e "source \$VIRTUAL_ENV/bin/geant4.sh" >> $VIRTUAL_ENV/bin/activate

## step 6: g4py
echo "setting up g4py... "
cd $VIRTUAL_ENV/src
hg clone $G4PY_REPOSITORY
cd g4py
export CLHEP_BASE_DIR=$VIRTUAL_ENV
# select system name from linux, linux64, macosx as appropriate
./configure linux64 --prefix=$VIRTUAL_ENV --with-g4-incdir=$VIRTUAL_ENV/include/Geant4 --with-g4-libdir=$VIRTUAL_ENV/lib --libdir=$VIRTUAL_ENV/lib/python2.7/site-packages/
if [ ! $? ]
then
  echo "error configuring g4py"
  exit 1
fi
if ! make -j$ncpus install
then
  echo "error building/installing g4py"
  exit 1
fi
echo "OK"

## step 7: chroma
echo "setting up chroma... "
if [ ! -f $HOME/.aksetup-defaults.py ]
then
  echo "BOOST_INC_DIR = ['/usr/include/boost']" > $HOME/.aksetup-defaults.py
  echo "BOOST_LIB_DIR = ['/usr/lib64']" >> $HOME/.aksetup-defaults.py
  echo "BOOST_PYTHON_LIBNAME = ['boost_python-mt-py27']" >> $HOME/.aksetup-defaults.py
  echo "created $HOME/.aksetup-defaults.py"
fi
if ! pip install -U distribute
then
  echo "error installing package distribute"
  exit 1
fi

if ! pip install pyublas
then
  echo "error installing package pyublas"
  exit 1
fi

# Bug workaround for Numpy 1.6.1
mkdir $VIRTUAL_ENV/local
ln -s $VIRTUAL_ENV/lib $VIRTUAL_ENV/local/lib

if ! pip install -e $CHROMA_URL
then
  echo "error installing package chroma"
  exit 1
fi

echo "OK"

## done
source $VIRTUAL_ENV/bin/activate
echo "chroma setup completed successfully"
echo "to use, run 'source $envname/bin/activate'"

