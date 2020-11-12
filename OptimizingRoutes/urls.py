"""OptimizingRoutes URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth.models import Group
from django.conf import settings
from django.conf.urls.static import static

from mainpage_routes import views as main_views
from login_routes.views import login_view
from django.contrib.auth import views as auth_views

from django.views.static import serve
from django.conf.urls import url

admin.site.site_header = 'Adminisración del portal de rutas'
admin.site.site_title = 'Gestión de pedidos y rutas'
admin.site.index_title = 'Administrador del portal de rutas'

admin.site.unregister(Group)

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', main_views.index, name='index'),

    path('login/', login_view, name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='logout.html'), name='logout'),

    path('routes/', main_views.map_mainpage, name='map'),
    path('routes/calculate', main_views.route_calculator, name='calculator'),

    url(r'^media/(?P<path>.*)$', serve,{'document_root':       settings.MEDIA_ROOT}), 
    url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}), 
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
