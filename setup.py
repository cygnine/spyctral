#!/usr/bin/env python
# -*- coding: latin-1 -*-

# Spyctrl - the Hybrid'n'Easy DG Environment
# Copyright (C) 2009 Akil C. Narayan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




import ez_setup
ez_setup.use_setuptools()
from setuptools import setup




def main():
    import glob

    setup(name="spyctral",
            # metadata
            version="0.5",
            description="Spectral Methods in 1D",
            #long_description="""
            #""",
            author=u"Akil Narayan",
            author_email="anaray@dam.brown.edu",
            license = "GPLv3",
            url="http://pypi.python.org/pypi/spyctral",
            classifiers=[
              'Environment :: Console',
              'Development Status :: 4 - Beta',
              'Intended Audience :: Developers',
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License (GPL)',
              'Natural Language :: English',
              'Programming Language :: C++',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Physics',
              ],

            # build info
            packages=[
                    "spyctral",
                    "spyctral.fourier",
                    "spyctral.opoly1d",
                    "spyctral.wienerfun",
                    ],
            zip_safe=False,
            )




if __name__ == '__main__':
    main()
