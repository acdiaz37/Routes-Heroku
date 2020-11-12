from django import forms
from django.contrib.auth import authenticate
from django.contrib.auth.forms import ReadOnlyPasswordHashField

from .models import Users


class UserLoginForm(forms.Form):

    username = forms.CharField(
        label='Usuario',
        widget=forms.TextInput(
            attrs={
                'type': 'email',
                'class': 'input100',
                'name': 'username',
                'placeholder': 'Correo electrónico',
            }
        ),
        required=True
    )

    password = forms.CharField(
        label='Contraseña',
        widget=forms.PasswordInput(
            attrs={
                'type': 'password',
                'class': 'input100',
                'name': 'pass',
                'placeholder': 'Contraseña',
            }
        ),
        required=True
    )

    def clean(self, *args, **kwargs):

        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username and password:

            user = authenticate(username=username, password=password)

            if not user:
                raise forms.ValidationError('Por favor verifique los datos de usuario ingresados.')
            if not user.check_password(password):
                raise forms.ValidationError('Por favor verifique la contraseña ingresada.')
            if not user.is_active:
                raise forms.ValidationError('El usuario ingresado no está activo.')

        return super(UserLoginForm, self).clean()


# Form de registro de usuarios en el panel administrativo

class UserAdminCreationForm(forms.ModelForm):

    email = forms.EmailField(label='Correo electrónico')
    first_name = forms.CharField(max_length=100, required=True)
    last_name = forms.CharField(max_length=100, required=True)
    cedula = forms.CharField(widget=forms.NumberInput, max_length=10, required=True)
    licencia = forms.CharField(widget=forms.NumberInput, max_length=15, required=True)

    password1 = forms.CharField(label='Contraseña', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Verifique su contraseña', widget=forms.PasswordInput)

    class Meta:

        model = Users
        fields = [
            'email',
            'password1',
            'password2',
            'first_name',
            'last_name',
            'cedula',
            'licencia'
        ]

    def clean_password2(self):

        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')

        if password1 and password2 and password1 != password2:
            raise forms.ValidationError('Verifique la contraseña ingresada')

        return password2

    def save(self, commit=True):

        password2 = self.cleaned_data.get('password2')

        user = super(UserAdminCreationForm, self).save(commit=False)
        user.set_password(password2)

        if commit:
            user.save()

        return user


# Form de cambios a usuarios en el panel administrativo

class UserAdminChangeForm(forms.ModelForm):

    password = ReadOnlyPasswordHashField()

    class Meta:

        model = Users
        fields = [
            'email',
            'password',
            'first_name',
            'last_name',
            'cedula',
            'licencia',
            'active',
            'admin'
        ]

    def clean_password(self):
        return self.initial['password']
