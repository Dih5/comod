"""comod - Compartment model Python package"""

__version__ = '0.2.0'
__author__ = 'Dih5 <dihedralfive@gmail.com>'

from .base import Model, FunctionModel, add_natural
from .community import CommunityModel
from .models import sir, seir, sirs, sis
