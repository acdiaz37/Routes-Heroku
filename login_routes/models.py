import secrets

from django.db import models

from django.core.exceptions import ValidationError

from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager
)


# Ciudades designadas para registros
class Ciudad(models.Model):

    nombre = models.CharField(
        verbose_name='Nombre de la ciudad',
        help_text='Indique el nombre de la ciudad a registrar',
        max_length=50,
        blank=True
    )

    REQUIRED_FIELDS = [
        'nombre'
    ]

    class Meta:
        verbose_name = 'Ciudad'
        verbose_name_plural = 'Ciudades'

    def __str__(self):
        return self.nombre


# Modelo para gestión de usuarios en Django y SQLite

class UserManager(BaseUserManager):

    def create_user(self, email, first_name, last_name, cedula, licencia,
                    password=None, is_active=True, is_staff=False, is_admin=False):

        if not email:
            raise ValueError('Es requerido ingresar el correo electrónico.')
        if not password:
            raise ValueError('Es requerido ingresar una contraseña.')

        user_obj = self.model(
            email=self.normalize_email(email),

            first_name=first_name,
            last_name=last_name,
            cedula=cedula,
            licencia=licencia,
        )
        user_obj.set_password(password)
        user_obj.active = is_active
        user_obj.admin = is_admin
        user_obj.staff = is_staff
        user_obj.save(using=self._db)

        return user_obj

    def create_staffuser(self, email, first_name, last_name, cedula, licencia, password=None):

        user = self.create_user(
            email=self.normalize_email(email),

            first_name=first_name,
            last_name=last_name,
            cedula=cedula,
            licencia=licencia,
            password=password,

            is_staff=True
        )

        return user

    def create_superuser(self, email, first_name, last_name, cedula, licencia, password=None):

        user = self.create_user(
            email=self.normalize_email(email),

            first_name=first_name,
            last_name=last_name,
            cedula=cedula,
            licencia=licencia,
            password=password,

            is_staff=True,
            is_admin=True
        )

        return user


def only_digits(value):
    if not value.isdigit():
        raise ValidationError('Este campo solo puede tener dígitos')


def random_values(x, y):
    secret_otp = secrets.SystemRandom()
    return str(secret_otp.randrange(x, y))


class Users(AbstractBaseUser):

    email = models.EmailField(
        verbose_name='Correo electrónico',
        help_text='Ingrese el correo electrónico de su cuenta',
        unique=True
    )
    first_name = models.CharField(
        verbose_name='Nombre',
        help_text='Indique su nombre',
        max_length=100
    )
    last_name = models.CharField(
        verbose_name='Apellidos',
        help_text='Indique sus apellidos',
        max_length=100
    )
    cedula = models.CharField(
        verbose_name='Número de cédula',
        help_text='Indique su número de cédula o licencia de conducir para su identificación en el sistema',
        max_length=10,
        unique=True,
        validators=[only_digits],
        default=random_values(1111111111, 9999999999)
    )
    licencia = models.CharField(
        verbose_name='Número de licencia de conducción',
        help_text='Indique su número de licencia de conducir para su identificación en el sistema',
        max_length=15,
        unique=True,
        validators=[only_digits],
        default=random_values(111111111111111, 999999999999999)
    )

    # atributos adicionales para el usuario

    active = models.BooleanField(
        verbose_name='¿Cuenta activa?',
        help_text='Especifique el estado activo de este usuario en el portal',
        default=True
    )
    staff = models.BooleanField(
        verbose_name='¿Personal del sitio?',
        help_text='Clarifique si este usuario hace parte del staff del portal',
        default=True
    )
    admin = models.BooleanField(
        verbose_name='¿Administrador del portal?',
        help_text='Indique si este usuario tiene privilegios de superusuario en el portal',
        default=False
    )

    objects = UserManager()

    USERNAME_FIELD = 'email'

    REQUIRED_FIELDS = [
        'first_name',
        'last_name',
        'cedula',
        'licencia'
    ]

    class Meta:
        verbose_name = 'Conductor'
        verbose_name_plural = 'Conductores'

    def __str__(self):
        return '[' + self.cedula + '] - ' + self.email

    @staticmethod
    def has_perm(perm, obj=None):
        print(perm, obj)
        return True

    @staticmethod
    def has_module_perms(app_label):
        print(app_label)
        return True

    @property
    def is_active(self):
        return self.active

    @property
    def is_admin(self):
        return self.admin

    @property
    def is_staff(self):
        return self.staff
