import numpy as np
import odl
from . import odl_torch
from utils import config

def getScanParam():
    args = config.getParse()
    space = odl.uniform_discr([-args.nSize/2.0*args.fPixelSize,-args.nSize/2.0*args.fPixelSize],
                              [args.nSize/2.0*args.fPixelSize,args.nSize/2.0*args.fPixelSize],[args.nSize,args.nSize],dtype='float32')
    angle_partition = odl.uniform_partition(0,2*np.pi,args.nViews)
    detector_partition = odl.uniform_partition(-args.nBins/2.0*args.fCellSize,args.nBins /2.0*args.fCellSize,args.nBins)
    geom = None
    if args.scanType == 'FanBeam':
        geom = odl.tomo.FanFlatGeometry(angle_partition,detector_partition,args.fSod,args.fOdd)
    elif args.scanType == 'Parallel':
        geom = odl.tomo.Parallel2dGeometry(angle_partition,detector_partition)
    return geom,space

def getForwardOperator(geom,space):
    forwardOperator = odl.tomo.RayTransform(space,geom,impl='astra_cuda')
    # opnorm = odl.power_method_opnorm(forwardOperator)
    # forwardOperator = (1 / opnorm) * forwardOperator
    return odl_torch.OperatorModule(forwardOperator)

def getBackWardOperator(geom,space):
    forwardOperator = odl.tomo.RayTransform(space, geom, impl='astra_cuda')
    opnorm = odl.power_method_opnorm(forwardOperator)
    backwardOperator = (1 / opnorm)**1 * forwardOperator.adjoint
    return odl_torch.OperatorModule(backwardOperator)