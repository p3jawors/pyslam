#!/usr/bin/env bash


# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

print_blue '================================================'
print_blue "Building Thirdparty"
print_blue '================================================'

set -e

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

# ====================================================
# N.B.: this script requires that you have first run:
#./install_basic.sh 
# ====================================================

echo '================================================'
print_blue "Configuring and building thirdparty/Pangolin ..."

make_dir thirdparty

INSTALL_PANGOLIN_ORIGINAL=0
cd thirdparty
if [ $INSTALL_PANGOLIN_ORIGINAL -eq 1 ] ; then
    # N.B.: pay attention this will generate a module 'pypangolin' ( it does not have the methods dcam.SetBounds(...) and pangolin.DrawPoints(points, colors)  )
    if [ ! -d pangolin ]; then
        sudo apt-get install -y libglew-dev
        git clone https://github.com/stevenlovegrove/Pangolin.git pangolin
        cd pangolin
        git submodule init && git submodule update
        cd ..
    fi
    cd pangolin
    make_dir build 
    if [ ! -f build/src/libpangolin.so ]; then
        cd build
        cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON $EXTERNAL_OPTION
        make -j8
        cd build/src
        ln -s pypangolin.*-linux-gnu.so  pangolin.linux-gnu.so
    fi
else
    # N.B.: pay attention this will generate a module 'pangolin' 
    if [ ! -d pangolin ]; then
        sudo apt-get install -y libglew-dev
        git clone https://github.com/uoip/pangolin.git
        cd pangolin
        PANGOLIN_UOIP_REVISION=3ac794a
        git checkout $PANGOLIN_UOIP_REVISION
        cd ..        
    fi
    cd pangolin
    if [ ! -f pangolin.cpython-*-linux-gnu.so ]; then   
        make_dir build   
        cd build
        cmake .. -DBUILD_PANGOLIN_LIBREALSENSE=OFF # disable realsense 
        make -j8
        cd ..
        #python setup.py install
    fi
fi
cd $STARTING_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/g2o ..."

cd thirdparty
if [ ! -d g2opy ]; then
    sudo apt-get install -y libsuitesparse-dev libeigen3-dev
	git clone https://github.com/uoip/g2opy.git
    cd g2opy
    G2OPY_REVISION=5587024
    git checkout $G2OPY_REVISION
    cd ..
    # copy local changes 
    cp ./g2opy_changes/types_six_dof_expmap.h ./g2opy/python/types/sba/types_six_dof_expmap.h
    cp ./g2opy_changes/sparse_optimizer.h ./g2opy/python/core/sparse_optimizer.h    
fi
cd g2opy
if [ ! -f thirdparty/g2o.cpython-*-linux-gnu.so ]; then  
    make_buid_dir
    cd build
    cmake ..
    make -j8
    cd ..
    #python3 setup.py install --user
fi    
cd $STARTING_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/orbslam2_features ..."
cd thirdparty/orbslam2_features
. build.sh
cd $STARTING_DIR

